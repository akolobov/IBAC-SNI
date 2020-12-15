"""
This is a copy of PPO from openai/baselines (https://github.com/openai/baselines/blob/52255beda5f5c8760b0ae1f676aa656bb1a61f80/baselines/ppo2/ppo2.py) with some minor changes.
"""

import time
import joblib
import numpy as np
import tensorflow as tf
from collections import deque
import gym3
from PIL import Image
import ffmpeg
import datetime
# from keras import backend as K
# from keras.engine.topology import Layer
#from tensorflow.keras.layers import Flatten




from mpi4py import MPI

from coinrun.tb_utils import TB_Writer
import coinrun.main_utils as utils

from coinrun.config import Config

mpi_print = utils.mpi_print

import subprocess
import sys
import os
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
from tensorflow.python.ops.ragged.ragged_util import repeat

from random import choice

# helper function to turn numpy array into video file
def vidwrite(filename, images, framerate=60, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(filename, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


def get_avg_rep_vec(state, env, a_n, model, weight=1, dropout=False):
    """ Implementation of the 'averaging representation function', which finds the average latent vector
    of the successor states for a given state, giving an action invariant representation
    Args:
        states: List containing the ground truth state for each observation in the current minibatch
        env: current environment which is a dummy wrapper around a gym3 environment, from FakeEnv
            class which is defined in train_agent.py
        a_n: Integer. Used to define a range [0, a_n] actions
            can take.
        model: Custom Model object containing the current PPO model. Used to obtain the latent vectors from observations
        weight: float from 0 to 1 determining how much we weight the phi_bar vector before returning
        dropout: Boolean indicating whether we want to "dropout" our latent augmentation and replace it with a random standard normal
        vector instead. Useful for avoiding overfitting to the initial state distribution. 
    Returns:
        phi_bar: the average of the latent vectors for all the successor states of our input state
    """
    if dropout:
        # sample bernoulli 
        bernoulli = np.random.binomial(1, 0.5)
        if bernoulli == 1:
            phi_bar = np.zeros((Config.NUM_ENVS, Config.NUM_STEPS))
            for a in range(a_n):
                # sample o_i ~ T(s, a_i)
                env.callmethod("set_state", state)
                o_i, _, _, _ = env.step(np.ones((Config.NUM_ENVS,))*a)
                phi_i = model.rep_vec(o_i)
                phi_bar += phi_i/a_n
        else:
            phi_bar = np.random.normal(size=(Config.NUM_ENVS, Config.NUM_STEPS))
    else:
        env.callmethod("set_state", state)
        o_i, _, _, _ = env.step(np.ones((Config.NUM_ENVS,))*4)
        phi_bar = model.rep_vec(o_i)
        # for _ in range(a_n):
        #     # sample o_i ~ T(s, a_i)
        #     env.callmethod("set_state", state)
        #     o_i, _, _, _ = env.step(np.ones((Config.NUM_ENVS,))*4)
        #     phi_i = model.rep_vec(o_i)
        #     phi_bar += phi_i/a_n
    return phi_bar*weight


def get_latents_and_acts(state, env, a_n, model):
    """ Helper function to compute the ground truth latent vectors for successor states
    from a fixed state, as well as track the actions taken to reach those states.
    Args:
        state: List containing the ground truth state for the current each observation
        env: current environment which is a dummy wrapper around a gym3 environment, from FakeEnv
            class which is defined in train_agent.py
        a_n: Integer. Used to define a range [0, a_n] actions
            can take.
        model: Custom Model object containing the current PPO model. Used to obtain the latent vectors from observations
    Returns:
        phi_sp: the latent vector concat with the actions to reach them from s for all the successor states of our input state
    """
    phi_sp_list = []
    rewards_per_env = np.empty(shape=(1,))
    for a in range(a_n):
            # sample o_i ~ T(s, a_i)
            env.callmethod("set_state", state)
            o_i, rewards, dones, infos = env.step(np.ones((Config.NUM_ENVS,))*a)
            curr_reward = np.average(rewards)
            rewards_per_env = np.append(arr=rewards_per_env, values=curr_reward,)
            phi_i = model.rep_vec(o_i)
            
            acts = np.ones(shape=(Config.NUM_ENVS, 1))*a
            phi_i = np.concatenate([phi_i, acts], axis=1)
            
            # (Num_envs, Latent_Dim)
            phi_sp_list.append(phi_i)


    reward = np.argmax(rewards_per_env)
    # (num_envs, latent_dim)
    phi_sps = phi_sp_list[reward-1]
    return phi_sps

class MpiAdamOptimizer(tf.compat.v1.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, **kwargs):
        self.comm = comm
        self.train_frac = 1.0 - Config.get_test_frac()
        tf.compat.v1.train.AdamOptimizer.__init__(self, **kwargs)
    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.compat.v1.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)

        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0)

        if Config.is_test_rank():
            flat_grad = tf.zeros_like(flat_grad)

        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(sum(sizes), np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks) * self.train_frac, out=buf)
            return buf

        avg_flat_grad = tf.compat.v1.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

# cosine similarity from SimSiam where
# 'z' is the stopgraded element. In our case
# this is the ground truth average next state
# latent vector.
def cos_loss(p, z):
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return -tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=(p*z), axis=1))

# augment with action takes in an array of latent vectors and augments them with
# each of the 15 possible actions in procgen.
def aug_w_acts(latents, nbatch):
    # assume latents are (1024, 256)
    latents_list = []
    for a in range(15):
        acts = np.ones((nbatch, 1))*a
        # (1024, 257)
        curr_latent = tf.concat((latents, acts), axis=1)
        latents_list.append(curr_latent)
    latents_list = tf.stack(latents_list)
    return latents_list


# same as above except vectorized
#def aug_w_acts(latents, nbatch):


# bilinear prediction layer 
def get_predictor():
    inputs = tf.keras.layers.Input((257, ))
    #x = tf.keras.layers.Dense(512, activation='relu', use_bias=False)(inputs)
    #x = tf.keras.layers.BatchNormalization()(x)
    p = tf.keras.layers.Dense(256)(inputs)

    h = tf.keras.Model(inputs, p)

    return h

# class BilinearModel(Layer):
#     """
#     Weighted bilinear model of two inputs. Useful for learning a model of linear interactions between
#     separate feature types (e.g., texture X spatial) or scales (e.g., dense X dilated), etc.
#     # taken from: https://github.com/rgmyr/keras-texture/blob/master/texture/layers/bilinearmodel.py?fbclid=IwAR1LtTYPjgKj-dI2Z8BvIzvP7VLZ30HAcOW-tuZILVjGunbhD8LhmJtzfI8
#     """

#     def __init__(self, l2_normalize=True, **kwargs):
#         self.l2_normalize = l2_normalize
#         super(BilinearModel, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self._shapecheck(input_shape)

#         self.shapeA, self.shapeB = input_shape[0][1], input_shape[1][1]

#         self.weights = self.add_weight(name='outer_prod_weights',
#                                        shape=(self.shapeA, self.shapeB),
#                                        initializer='glorot_normal',
#                                        trainable=True)
#     def call(self, x):
#         self._shapecheck(x)

#         weighted_outer = tf.multiply(self.weights, tf.einsum('bi,bj->bij', x[0], x[1]))

#         flat_output = K.Flatten(weighted_outer)

#         if self.l2_normalize:
#             flat_output = tf.nn.l2_normalize(flat_output, axis=-1)

#         return flat_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.shapeA, self.shapeB)

    def _shapecheck(self,x):
        # both x and input_shape should be a list of len=2
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError('A `BilinearModel` layer should be called on a list of exactly two inputs')

        # if input_shape, check dimensionality
        if isinstance(x[0], tuple):
            assert len(x[0]) == 2 and len(x[1]) == 2, '`BilinearModel` input tensors must be 2-D'

        # if x, they should match shapes from build()
        elif K.is_keras_tensor(x[0]):
            assert K.ndim(x[0]) == 2 and K.ndim(x[1]) == 2, '`BilinearModel` input tensors must be 2-D'
            shapeA, shapeB = K.int_shape(x[0])[1], K.int_shape(x[1])[1]
            if shapeA != self.shapeA or shapeB != self.shapeB:
                raise ValueError('Unexpected `BilinearModel` input_shape')

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        self.max_grad_norm = max_grad_norm

        sess = tf.compat.v1.get_default_session()

        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, max_grad_norm)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, max_grad_norm)

        # in case we don't use rep loss
        rep_loss = None
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.compat.v1.placeholder(tf.float32, [None])
        R = tf.compat.v1.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [None])
        OLDVPRED = tf.compat.v1.placeholder(tf.float32, [None])
        LR = tf.compat.v1.placeholder(tf.float32, [])
        CLIPRANGE = tf.compat.v1.placeholder(tf.float32, [])
        # VF loss
        vpred = train_model.vf_train  # Same as vf_run for SNI and default, but noisy for SNI2 while the boostrap is not
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf_train - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1, vf_losses2))

        neglogpac_train = train_model.pd_train.neglogp(A)
        ratio_train = tf.exp(OLDNEGLOGPAC - neglogpac_train)
        pg_losses_train = -ADV * ratio_train
        pg_losses2_train = -ADV * tf.clip_by_value(ratio_train, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(input_tensor=tf.maximum(pg_losses_train, pg_losses2_train))
        approxkl_train = .5 * tf.reduce_mean(input_tensor=tf.square(neglogpac_train - OLDNEGLOGPAC))
        clipfrac_train = tf.reduce_mean(input_tensor=tf.cast(tf.greater(tf.abs(ratio_train - 1.0), CLIPRANGE), dtype=tf.float32))

        if Config.BETA >= 0:
            entropy = tf.reduce_mean(input_tensor=train_model.pd_train._components_distribution.entropy())
        else:
            entropy = tf.reduce_mean(input_tensor=train_model.pd_train.entropy())

        # Add entropy and policy loss for the samples as well
        if Config.SNI or Config.SNI2:
            neglogpac_run = train_model.pd_run.neglogp(A)
            ratio_run = tf.exp(OLDNEGLOGPAC - neglogpac_run)
            pg_losses_run = -ADV * ratio_run
            pg_losses2_run = -ADV * tf.clip_by_value(ratio_run, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

            pg_loss += tf.reduce_mean(input_tensor=tf.maximum(pg_losses_run, pg_losses2_run))
            pg_loss /= 2.

            entropy += tf.reduce_mean(input_tensor=train_model.pd_run.entropy())
            entropy /= 2.

            approxkl_run = .5 * tf.reduce_mean(input_tensor=tf.square(neglogpac_run - OLDNEGLOGPAC))
            clipfrac_run = tf.reduce_mean(input_tensor=tf.cast(tf.greater(tf.abs(ratio_run - 1.0), CLIPRANGE), dtype=tf.float32))
        else:
            approxkl_run = tf.constant(0.)
            clipfrac_run = tf.constant(0.)

        # custom rep loss
        if Config.CUSTOM_REP_LOSS:
            REP_PROC = tf.compat.v1.placeholder(dtype=tf.float32, shape=(nbatch_train, 257))
            pred_h = get_predictor()
            acts = REP_PROC[:, 255:256]
            
            aug_h = tf.concat([train_model.h, acts], axis=1)
            
            # (nbatch_train, 257)
            pred_latents = pred_h(aug_h)

            pred_latents = tf.reshape(pred_latents, [-1, 256])

            # (nbatch_train, 256)
            phi_s = REP_PROC[:, :-1]
    
            rep_loss = cos_loss(pred_latents,  phi_s)

        # cosine similarity loss

        params = tf.compat.v1.trainable_variables()
        weight_params = [v for v in params if '/b' not in v.name]

        total_num_params = 0

        for p in params:
            shape = p.get_shape().as_list()
            num_params = np.prod(shape)
            mpi_print('param', p, num_params)
            total_num_params += num_params

        mpi_print('total num params:', total_num_params)

        l2_loss = tf.reduce_sum(input_tensor=[tf.nn.l2_loss(v) for v in weight_params])

        # The first occurance should be in the train_model

        if Config.BETA >= 0:
            info_loss = tf.compat.v1.get_collection(
                key="INFO_LOSS",
                scope="model/info_loss"
            )
            beta = Config.BETA

        elif Config.BETA_L2A >= 0:
            info_loss = tf.compat.v1.get_collection(
                key="INFO_LOSS_L2A",
                scope="model/info_loss"
            )
            beta = Config.BETA_L2A
        else:
            info_loss = [tf.constant(0.)]
            beta = 0

        print(info_loss)
        assert len(info_loss) == 1
        info_loss = info_loss[0]

        if Config.CUSTOM_REP_LOSS:
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * Config.L2_WEIGHT + beta * info_loss + rep_loss*0.0001
        else:
            loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * Config.L2_WEIGHT + beta * info_loss

        if Config.SYNC_FROM_ROOT:
            trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        
        self.opt = trainer
        grads_and_var = trainer.compute_gradients(loss, params)

        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        _train = trainer.apply_gradients(grads_and_var)

        
        
        def train(lr, cliprange, obs, phi_bars, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            adv_mean = np.mean(advs, axis=0, keepdims=True)
            adv_std = np.std(advs, axis=0, keepdims=True)
            advs = (advs - adv_mean) / (adv_std + 1e-8)

            if Config.CUSTOM_REP_LOSS:
                td_map = {train_model.X:obs, REP_PROC:phi_bars, A:actions, ADV:advs, R:returns, LR:lr,
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            else:
                td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                        CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            
            if Config.CUSTOM_REP_LOSS:
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl_train, clipfrac_train, approxkl_run, clipfrac_run, l2_loss, info_loss, rep_loss, _train],
                    td_map
                )[:-1]
            else:
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl_train, clipfrac_train, approxkl_run, clipfrac_run, l2_loss, info_loss, _train],
                    td_map
                )[:-1]
            
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl_train', 'clipfrac_train', 'approxkl_run', 'clipfrac_run', 'l2_loss', 'info_loss_cv', 'rep_loss']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.rep_vec = act_model.rep_vec
        self.custom_train = train_model.custom_train

        if Config.SYNC_FROM_ROOT:
            if MPI.COMM_WORLD.Get_rank() == 0:
                initialize()
            
            global_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="")
            sess.run(tf.compat.v1.global_variables_initializer())
            sync_from_root(sess, global_variables) #pylint: disable=E1101
        else:
            initialize()

class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        # List of two element tuples containing state lists for procgen,
        # where each tuple is the start & ending state for a trajectory.
        # The intuition here is that start and ending states will be very
        # different, giving us good positive/negative examples.
        self.diff_states = []

    def run(self, update_frac):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = []
        epinfos = []
        mb_phi_bars = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            if Config.CUSTOM_REP_LOSS:
                # collect the ground truth state for each observation
                curr_state = self.env.callmethod("get_state")
                mb_states.append(curr_state)
                phi_bar = get_latents_and_acts(curr_state, self.env, self.env.action_space.n, self.model)
                # return environment to last state after sampling
                mb_phi_bars.append(phi_bar)
                self.env.callmethod("set_state", curr_state)
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, update_frac, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            
            
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        
        #batch of steps to batch of rollouts
        if Config.CUSTOM_REP_LOSS:
            mb_phi_bars = np.stack(mb_phi_bars)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        #vidwrite('plunder_avg_phi_attempt-{}.avi'.format(datetime.datetime.now().timestamp()), mb_obs[:, 0, :, :, :])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, update_frac, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_phi_bars, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
             vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    #tf.compat.v1.disable_v2_behavior()
    sess = tf.compat.v1.get_default_session()

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    
    nbatch_train = nbatch // nminibatches

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    utils.load_all_params(sess)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    tfirststart = time.time()
    active_ep_buf = epinfobuf100

    nupdates = total_timesteps//nbatch
    mean_rewards = []
    datapoints = []

    run_t_total = 0
    train_t_total = 0

    can_save = True
    checkpoints = [32, 64]
    saved_key_checkpoints = [False] * len(checkpoints)

    if Config.SYNC_FROM_ROOT and rank != 0:
        can_save = False

    def save_model(base_name=None):
        base_dict = {'datapoints': datapoints}
        utils.save_params_in_scopes(sess, ['model'], Config.get_save_file(base_name=base_name), base_dict)

    # For logging purposes, allow restoring of update
    start_update = 0
    if Config.RESTORE_STEP is not None:
        start_update = Config.RESTORE_STEP // nbatch

    tb_writer = TB_Writer(sess)
    for update in range(start_update+1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        mpi_print('collecting rollouts...')
        run_tstart = time.time()

        packed = runner.run(update_frac=update/nupdates)
        obs, returns, masks, actions, values, neglogpacs, phi_bars, epinfos = packed
        # reshape our augmented state vectors to match first dim of observation array
        # (mb_size*num_envs, latent_dim + one_action)
        if Config.CUSTOM_REP_LOSS:
            phi_bars = phi_bars.reshape(Config.NUM_STEPS*Config.NUM_ENVS, Config.NODES+1)
        avg_value = np.mean(values)
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed
        mpi_print('rollouts complete')

        mblossvals = []

        mpi_print('updating parameters...')
        train_tstart = time.time()

        mean_cust_loss = 0
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                sess.run([model.train_model.train_dropout_assign_ops])
                end = start + nbatch_train
                mbinds = inds[start:end]
                if Config.CUSTOM_REP_LOSS:
                    slices = (arr[mbinds] for arr in (obs, phi_bars, returns, masks, actions, values, neglogpacs))
                else:
                    # since we don't use phi_bars, use obs as dummy variable
                    dummy = obs
                    slices = (arr[mbinds] for arr in (obs, dummy, returns, masks, actions, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        # update the dropout mask
        sess.run([model.train_model.train_dropout_assign_ops])
        sess.run([model.train_model.run_dropout_assign_ops])

        train_elapsed = time.time() - train_tstart
        train_t_total += train_elapsed
        mpi_print('update complete')

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            step = update*nbatch
            rew_mean_10 = utils.process_ep_buf(active_ep_buf, tb_writer=tb_writer, suffix='', step=step)
            ep_len_mean = np.nanmean([epinfo['l'] for epinfo in active_ep_buf])
            
            mpi_print('\n----', update)

            mean_rewards.append(rew_mean_10)
            datapoints.append([step, rew_mean_10])
            tb_writer.log_scalar(ep_len_mean, 'ep_len_mean', step=step)
            tb_writer.log_scalar(fps, 'fps', step=step)
            tb_writer.log_scalar(avg_value, 'avg_value', step=step)
            tb_writer.log_scalar(mean_cust_loss, 'custom_loss', step=step)


            mpi_print('time_elapsed', tnow - tfirststart, run_t_total, train_t_total)
            mpi_print('timesteps', update*nsteps, total_timesteps)

            mpi_print('eplenmean', ep_len_mean)
            mpi_print('eprew', rew_mean_10)
            mpi_print('fps', fps)
            mpi_print('total_timesteps', update*nbatch)
            mpi_print([epinfo['r'] for epinfo in epinfobuf10])

            if len(mblossvals):
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    mpi_print(lossname, lossval)
                    tb_writer.log_scalar(lossval, lossname, step=step)
            mpi_print('----\n')

        if can_save:
            if save_interval and (update % save_interval == 0):
                save_model()

            for j, checkpoint in enumerate(checkpoints):
                if (not saved_key_checkpoints[j]) and (step >= (checkpoint * 1e6)):
                    saved_key_checkpoints[j] = True
                    save_model(str(checkpoint) + 'M')

    save_model()

    env.close()
    return mean_rewards

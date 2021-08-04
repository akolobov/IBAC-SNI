"""
This is a copy of PPO from openai/baselines (https://github.com/openai/baselines/blob/52255beda5f5c8760b0ae1f676aa656bb1a61f80/baselines/ppo2/ppo2.py) with some minor changes.
"""
import wandb
import time
import joblib
import numpy as np
import tensorflow as tf
from collections import deque
import gym3
from PIL import Image
import ffmpeg
import datetime




from mpi4py import MPI

from coinrun.tb_utils import TB_Writer
import coinrun.main_utils as utils

from coinrun.train_agent import make_env
from coinrun.config import Config, count_latent_factors

mpi_print = utils.mpi_print

import subprocess
import sys
import os
import copy
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
from tensorflow.python.ops.ragged.ragged_util import repeat

from random import choice

"""
Intrinsic advantage methods
"""

class RunningStats(object):
    # https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/RND_PPO/RND_PPO_cont_ftr_nsn_mtCar_php.ipynb
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        #self.std = np.sqrt(np.maximum(self.var, 1e-2))                            
        self.count = batch_count + self.count

def running_stats_fun(run_stats, buf, clip, clip_state):
    run_stats.update(np.array(buf))
    buf = (np.array(buf) - run_stats.mean) / run_stats.std   
    if clip_state == True:
      buf = np.clip(buf, -clip, clip)
    return buf


def soft_update(source_variables,target_variables,tau=1.0):
    for (v_s, v_t) in zip(source_variables, target_variables):
        v_t.shape.assert_is_compatible_with(v_s.shape)
        v_t.assign((1 - tau) * v_t + tau * v_s)

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


# prediction layer 
def get_seq_encoder():
    inputs = tf.keras.layers.Input((Config.POLICY_NHEADS,Config.REP_LOSS_M,256 ))
    conv1x1 = tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1),activation='relu')(inputs)
    out = tf.keras.layers.Dense(512,activation='relu')(tf.reshape(conv1x1,(Config.NUM_ENVS*Config.POLICY_NHEADS,Config.REP_LOSS_M*256)))
    # output should be (ne, N, x)
    p = tf.reshape(out,(Config.NUM_ENVS, Config.POLICY_NHEADS,-1))
    #x = tf.keras.layers.Dense(512, activation='relu', use_bias=False)(inputs)
    #x = tf.keras.layers.BatchNormalization()(x)
    # p = tf.keras.layers.Dense(271)(inputs)
    h = tf.keras.Model(inputs, p)
    return h

def get_anch_encoder():
    inputs = tf.keras.layers.Input((Config.POLICY_NHEADS,256 ))
    p = tf.keras.layers.Dense(512,activation='relu')(inputs)
    h = tf.keras.Model(inputs, p)
    return h

def get_predictor(n_out=512):
    inputs = tf.keras.layers.Input((512, ))
    p = tf.keras.layers.Dense(512,activation='relu')(inputs)
    p2 = tf.keras.layers.Dense(n_out)(p)
    h = tf.keras.Model(inputs, p2)
    return h


def tanh_clip(x, clip_val=20.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    Trick from AM-DIM
    '''
    if clip_val is not None:
        x_clip = clip_val * tf.math.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        self.max_grad_norm = max_grad_norm
        self.head_idx_current_batch = 0
        self.critic_idx_current_batch = 0
        sess = tf.compat.v1.get_default_session()

        self.running_stats_s = RunningStats()
        self.running_stats_s_ = RunningStats()
        self.running_stats_r = RunningStats()
        self.running_stats_r_i = RunningStats()

        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, max_grad_norm)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, max_grad_norm)
        self.train_model = train_model
        # in case we don't use rep loss
        rep_loss = None
        # HEAD_IDX = tf.compat.v1.placeholder(tf.int32, [None])
        A = train_model.pdtype.sample_placeholder([None],name='A')
        A_i = train_model.A_i
        LATENT_FACTORS = train_model.pdtype.sample_placeholder([Config.REP_LOSS_M,Config.POLICY_NHEADS,None,count_latent_factors(Config.ENVIRONMENT)],name='LATENT_FACTORS')
        ADV = tf.compat.v1.placeholder(tf.float32, [None],name='ADV')
        R = tf.compat.v1.placeholder(tf.float32, [None],name='R')
        R_NCE = tf.compat.v1.placeholder(tf.float32, [Config.REP_LOSS_M,1,None],name='R_NCE')
        OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [None],name='OLDNEGLOGPAC')
        OLDNEGLOGPAC_i = tf.compat.v1.placeholder(tf.float32, [None],name='OLDNEGLOGPAC_i')
        LR = tf.compat.v1.placeholder(tf.float32, [],name='LR')
        CLIPRANGE = tf.compat.v1.placeholder(tf.float32, [],name='CLIPRANGE')

        if Config.CUSTOM_REP_LOSS:
            ADV_i= tf.compat.v1.placeholder(tf.float32, [None])
            R_i = tf.compat.v1.placeholder(tf.float32, [None])
            OLDVPRED_i = tf.compat.v1.placeholder(tf.float32, [None])
            vpred_i = train_model.vf_i_train  # Same as vf_run for SNI and default, but noisy for SNI2 while the boostrap is not
            vpredclipped_i = OLDVPRED_i + tf.clip_by_value(vpred_i - OLDVPRED_i, - CLIPRANGE, CLIPRANGE)
            vf_losses1_i = tf.square(vpred_i - R_i)
            vf_losses2_i = tf.square(vpredclipped_i - R_i)
            vf_loss_i = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1_i, vf_losses2_i))

            # ADV = ADV + ADV_i

        # TD loss for critic
        # VF loss
        OLDVPRED = tf.compat.v1.placeholder(tf.float32, [None],name='OLDVPRED')
        vpred = train_model.vf_train  # Same as vf_run for SNI and default, but noisy for SNI2 while the boostrap is not
        if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
            vpred = vpred[self.critic_idx_current_batch]
        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1, vf_losses2))

        neglogpac_train = train_model.pd_train[0].neglogp(A)
        ratio_train = tf.exp(OLDNEGLOGPAC - neglogpac_train)
        pg_losses_train = -ADV * ratio_train
        pg_losses2_train = -ADV * tf.clip_by_value(ratio_train, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(input_tensor=tf.maximum(pg_losses_train, pg_losses2_train))
        approxkl_train = .5 * tf.reduce_mean(input_tensor=tf.square(neglogpac_train - OLDNEGLOGPAC))
        clipfrac_train = tf.reduce_mean(input_tensor=tf.cast(tf.greater(tf.abs(ratio_train - 1.0), CLIPRANGE), dtype=tf.float32))

        if Config.CUSTOM_REP_LOSS:
            neglogpac_train_i = train_model.pd_train_i.neglogp(A_i[:,0,self.head_idx_current_batch])
            ratio_train_i = tf.exp(OLDNEGLOGPAC_i - neglogpac_train_i)
            pg_losses_train_i = -ADV_i * ratio_train_i
            pg_losses2_train_i = -ADV_i * tf.clip_by_value(ratio_train_i, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss_i = tf.reduce_mean(input_tensor=tf.maximum(pg_losses_train_i, pg_losses2_train_i))
        else:
            pg_loss_i = tf.constant(0.,dtype=tf.float32)

        if Config.BETA >= 0:
            entropy = tf.reduce_mean(input_tensor=train_model.pd_train[0]._components_distribution.entropy())
        else:
            entropy = tf.reduce_mean(input_tensor=train_model.pd_train[0].entropy())

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

        # print(info_loss)
        assert len(info_loss) == 1
        info_loss = info_loss[0]

        if Config.CUSTOM_REP_LOSS:
            rep_loss = tf.reduce_mean(train_model.rep_loss)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * Config.L2_WEIGHT + beta * info_loss

        if Config.SYNC_FROM_ROOT:
            trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
            trainer_encoder = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
            trainer_latent_transition = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        
        self.opt = trainer
        grads_and_var = trainer.compute_gradients(loss, params)


        
        
        grads, var = zip(*grads_and_var)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        tot_norm = tf.zeros((1,))
        for g,v in grads_and_var:
            tot_norm += tf.norm(g)
        tot_norm = tf.reshape(tot_norm, [])

        _train = trainer.apply_gradients(grads_and_var)


        grads_and_var_encoder = trainer_encoder.compute_gradients(train_model.encoder_bisimilarity_loss, params)
        grads_encoder, var_encoder = zip(*grads_and_var_encoder)
        if max_grad_norm is not None:
            grads_encoder, _grad_norm = tf.clip_by_global_norm(grads_encoder, max_grad_norm)
        grads_and_var_encoder = list(zip(grads_encoder, var_encoder))
        _train_encoder = trainer_encoder.apply_gradients(grads_and_var_encoder)

        grads_and_var_latent = trainer_latent_transition.compute_gradients(train_model.latent_transition_loss, params)
        grads_latent, var_latent = zip(*grads_and_var_latent)
        if max_grad_norm is not None:
            grads_latent, _grad_norm = tf.clip_by_global_norm(grads_latent, max_grad_norm)
        grads_and_var_latent = list(zip(grads_latent, var_latent))
        _train_latent = trainer_latent_transition.apply_gradients(grads_and_var_latent)

        
        
        
        def train(lr, cliprange, obs, returns, masks, actions, infos, values, neglogpacs, rewards, train_target='policy'):
            values = values[:,self.critic_idx_current_batch] if Config.CUSTOM_REP_LOSS else values
            advs = returns - values
            adv_mean = np.mean(advs, axis=0, keepdims=True)
            adv_std = np.std(advs, axis=0, keepdims=True)
            advs = (advs - adv_mean) / (adv_std + 1e-8)

            if Config.CUSTOM_REP_LOSS:
                advs_i = returns_i - values_i
                
            
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, train_model.R:rewards, train_model.A:actions, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values
                    }
                    
            # import ipdb;ipdb.set_trace()
            if train_target == 'policy':
                rets = sess.run( [pg_loss, vf_loss, entropy, approxkl_train, clipfrac_train, approxkl_run, clipfrac_run, l2_loss, info_loss, tot_norm, _train],td_map)[:-1]
                return rets
            elif train_target == 'encoder':
                rets = sess.run( [_train_encoder],td_map)[:-1]
                return rets
            elif train_target == 'latent':
                rets = sess.run( [_train_latent],td_map)[:-1]
                return rets
            
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl_train', 'clipfrac_train', 'approxkl_run', 'clipfrac_run', 'l2_loss', 'info_loss_cv', 'rep_loss', 'value_i_loss', 'policy_loss_i','gradient_norm']

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
    def __init__(self, *, env, eval_env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)

        self.eval_obs = np.zeros((self.nenv,) + eval_env.observation_space.shape, dtype=eval_env.observation_space.dtype.name)
        self.eval_obs[:] = eval_env.reset()
        self.eval_dones = [False for _ in range(self.nenv)]

        self.lam = lam
        self.gamma = gamma
        # List of two element tuples containing state lists for procgen,
        # where each tuple is the start & ending state for a trajectory.
        # The intuition here is that start and ending states will be very
        # different, giving us good positive/negative examples.
        self.diff_states = []
        # create env just for resets
        # TODO: Set a Config variable that takes this in as arg
        total_timesteps = int(160e6)
        if Config.LONG_TRAINING:
            total_timesteps = int(200e6)
        elif Config.SHORT_TRAINING:
            #total_timesteps = int(120e6)
            total_timesteps = int(25e6)
        elif Config.VERY_SHORT_TRAINING:
            total_timesteps = int(5e6)
        self.reset_env = make_env(steps_per_env=total_timesteps//2)[0] # for _ in range(Config.POLICY_NHEADS)]
        self.eval_env = eval_env

    def get_NCE_samples(self, s_0, env,obs_0,done):
        states_nce = []
        rewards_nce = []
        actions_nce = []
        neglogps_nce = []
        dones_nce = []
        infos_nce = []
        v_is = []
        r_is = []
        k = 0
        obs = obs_0.copy()
        # for each policy head, collect m step rollouts
        env.callmethod("set_state", s_0)
        state_nce = []
        action_nce = []
        neglogp_nce = []
        reward_nce = []
        done_nce = []
        info_nce = []
        for m in range(Config.REP_LOSS_M):
            actions, values, _, neglogps = self.model.step(obs, 1)
            actions = actions[k]
            values = values[self.model.critic_idx_current_batch]
            neglogps = neglogps[k]
            obs, reward, done, info = env.step(actions)
            state_nce.append(obs)
            # if m == Config.REP_LOSS_M-1: # direct model for remaining of trajectory
            #     reward = reward + values
            action_nce.append(actions)
            neglogp_nce.append(neglogps)
            reward_nce.append(reward)
            done_nce.append(done)
            info_nce.append([[float(v) for k,v in info_.items() if (k != 'episode') and (Config.ENVIRONMENT in k)] for info_ in info])
        states_nce.append(state_nce)
        actions_nce.append(action_nce)
        neglogps_nce.append(neglogp_nce)
        rewards_nce.append(reward_nce)
        dones_nce.append(done_nce)
        infos_nce.append(info_nce)

        states_nce = np.transpose(np.array(states_nce),(1,0,2,3,4,5))
        actions_nce = np.transpose(np.array(actions_nce),(2,1,0))
        neglogps_nce = np.transpose(np.array(neglogps_nce),(1,0,2))
        rewards_nce = np.transpose(np.array(rewards_nce),(1,0,2))
        dones_nce = np.transpose(np.array(dones_nce),(1,0,2))
        infos_nce = np.transpose(np.array(infos_nce),(1,0,2,3))
        # each head learns on own samples
        self_labels = np.repeat(np.arange(0,Config.POLICY_NHEADS).reshape(-1,1),Config.NUM_ENVS,1)
        labels = self_labels
        
        v_i, r_i = self.model.train_model.nce_fw_pass(nce_dict={self.model.train_model.STATE_NCE:states_nce,self.model.train_model.ANCH_NCE:obs_0,self.model.train_model.LAB_NCE:labels,self.model.train_model.STATE:obs_0,self.model.train_model.A_i:actions_nce})
        # each head learns on quantiles of V
        # sum_rew = rewards_nce.sum(0)
        # quantiles = np.quantile(sum_rew,np.linspace(0,1,Config.POLICY_NHEADS+1))[1:]

        # if quantiles[0] == quantiles[-1]: # only 1 reward value -> train on own trajectory
        #     labels =  self_labels
        # else:
        #     labels = np.digitize(sum_rew, bins=quantiles)
        
        return states_nce, actions_nce, neglogps_nce, rewards_nce, dones_nce, infos_nce, labels, np.array(v_i), np.array(r_i)

    def run(self, update_frac):
        # print('Using head %d'%self.model.head_idx_current_batch)
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_infos = [],[],[],[],[],[],[]
        mb_states = []
        epinfos = []
        eval_epinfos = []

        mb_states_nce, mb_actions_nce, mb_neglogps_nce, mb_rewards_nce, mb_dones_nce, mb_infos_nce, mb_labels_nce, mb_rewards_i, mb_values_i, mb_anchors_nce = [], [], [], [], [], [], [], [], [], []

        states_nce, anchors_nce, labels_nce, rewards_nce, infos_nce = [],[],[],[],[]

        # ensure reset env has same step counter as main env
        self.reset_env.current_env_steps_left = self.env.current_env_steps_left
        # For n in range number of steps
        self.nce_update_freq = 8
        for t in range(self.nsteps):
            if Config.CUSTOM_REP_LOSS:
                # collect the ground truth state for each observation
                curr_state = self.env.callmethod("get_state")
                mb_states.append(curr_state)
                
                actions, values, self.states, neglogpacs = self.model.step(self.obs, update_frac)
                # if t == 0:
                #     # pi_weights = np.array(values).mean(1)
                #     # head_idx_current_batch = pi_weights.argmax() # do rollouts with head with largest rewards
                #     self.model.head_idx_current_batch = np.random.randint(0,Config.POLICY_NHEADS,1).item()
                if (t % self.nce_update_freq) == 0:
                    s_0 = self.env.callmethod("get_state")
                    anchors_nce = self.obs.copy()

                    states_nce, actions_nce, neglogps_nce, rewards_nce, dones_nce, infos_nce, labels_nce, rewards_i, values_i = self.get_NCE_samples(s_0, self.reset_env, anchors_nce, self.dones)
                    
                    # rewards_i = np.log(rewards_i**2+1) # as per RE3
                    # rewards_i = -rewards_i

                    mb_states_nce.append(states_nce)
                    mb_actions_nce.append(actions_nce)
                    mb_neglogps_nce.append(neglogps_nce)
                    mb_rewards_nce.append(rewards_nce)
                    mb_dones_nce.append(dones_nce)
                    mb_infos_nce.append(infos_nce)
                    mb_labels_nce.append(labels_nce)
                    mb_rewards_i.append(rewards_i)
                    mb_values_i.append(values_i)
                    mb_anchors_nce.append(anchors_nce)

                actions = actions[self.model.head_idx_current_batch]
                # values = values[head_idx_current_batch]
                neglogpacs = neglogpacs[self.model.head_idx_current_batch]


            else:
                # Given observations, get action value and neglopacs
                # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
                actions, values, self.states, neglogpacs = self.model.step(self.obs,  update_frac,None, self.dones)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, self.infos = self.env.step(actions)

            eval_actions, eval_values, eval_states, eval_neglogpacs = self.model.step(self.eval_obs, update_frac)
            if Config.CUSTOM_REP_LOSS:
                eval_actions = eval_actions[self.model.head_idx_current_batch]
            self.eval_obs[:], eval_rewards, self.eval_dones, self.eval_infos = self.eval_env.step(eval_actions)
        
            for info in self.infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            for info in self.eval_infos:
                eval_maybeepinfo = info.get('episode')
                if eval_maybeepinfo: eval_epinfos.append(eval_maybeepinfo)
        
            mb_infos.append([[float(v) for k,v in info_.items() if (k != 'episode') and (Config.ENVIRONMENT in k)] for info_ in self.infos])
            mb_rewards.append(rewards)
            
        # if Config.CUSTOM_REP_LOSS:
        #     s_0 = self.env.callmethod("get_state")
        #     anchors_nce = self.obs.copy()
        #     states_nce, rewards_nce, dones_nce, infos_nce, labels_nce, mb_rewards_i, mb_values_i = self.get_NCE_samples(s_0, self.reset_env, self.obs.copy(), self.dones)
        # else:
        #     states_nce = rewards_nce = dones_nce = infos_nce = labels_nce = anchors_nce = tf.compat.v1.placeholder(tf.float32, [None])

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        #vidwrite('plunder_avg_phi_attempt-{}.avi'.format(datetime.datetime.now().timestamp()), mb_obs[:, 0, :, :, :])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_rewards_i = np.asarray(mb_rewards_i, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_values_i = np.asarray(mb_values_i, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_infos = np.asarray(mb_infos, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        if Config.CUSTOM_REP_LOSS:
            mb_states_nce = np.asarray(mb_states_nce, dtype=np.float32).transpose(0,3,1,2,4,5,6)
            # mb_anchors_nce = np.asarray(mb_states_nce, dtype=np.float32).transpose(0,3,1,2,4,5,6)
            mb_actions_nce = np.asarray(mb_actions_nce, dtype=np.float32).transpose(0,3,1,2)
            mb_neglogps_nce = np.asarray(mb_neglogps_nce, dtype=np.float32).transpose(0,3,1,2)
            mb_rewards_nce = np.asarray(mb_rewards_nce, dtype=np.float32).transpose(0,3,1,2)
            mb_dones_nce = np.asarray(mb_dones_nce, dtype=np.float32).transpose(0,3,1,2)
            mb_anchors_nce = np.asarray(mb_anchors_nce, dtype=np.float32)
            if not len(mb_infos_nce):
                mb_infos_nce = np.zeros_like(mb_dones_nce)
            else:
                mb_infos_nce = np.asarray(mb_infos_nce, dtype=np.float32).transpose(0,3,1,2,4)
            mb_labels_nce = np.asarray(mb_labels_nce, dtype=np.float32).transpose(0,2,1)

        last_values = self.model.value(self.obs, update_frac, self.states, self.dones)[self.model.critic_idx_current_batch] #use first critic
        
        if Config.CUSTOM_REP_LOSS:
            last_values_i = self.model.train_model.value_i(self.obs, update_frac, self.states, self.dones)
            lastgaelam_i = 0
            mb_advs_i = np.zeros_like(mb_rewards_i)
            for t in range(len(mb_rewards_i)):
                mb_rewards_i[t] = running_stats_fun(self.model.running_stats_r_i, mb_rewards_i[t], 1, False)     
        
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
                nextvalues = mb_values[t+1][self.model.critic_idx_current_batch] if Config.CUSTOM_REP_LOSS else mb_values[t+1]
                
            if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t][self.model.critic_idx_current_batch]
            else:
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        if Config.CUSTOM_REP_LOSS:
            for t in reversed(range(len(mb_rewards_i))):
                """
                Intrinsic advantages
                """
                if t == len(mb_rewards_i) - 1:
                    nextnonterminal_i = 1.0 - self.dones
                    nextvalues_i = last_values_i
                else:
                    nextnonterminal_i = 1.0 - mb_dones[t+1]
                    nextvalues_i = mb_values_i[t+1]

                delta_i = mb_rewards_i[t] + self.gamma * nextvalues_i * nextnonterminal_i - mb_values_i[t]
                mb_advs_i[t] = lastgaelam_i = delta_i + self.gamma * self.lam * nextnonterminal_i * lastgaelam_i
        if Config.CUSTOM_REP_LOSS:
            mb_returns_i = mb_advs_i + mb_values_i
            mb_returns = mb_advs + mb_values[:,self.model.critic_idx_current_batch] # use first critic
        else:
            mb_returns = mb_advs + mb_values
            
        if Config.CUSTOM_REP_LOSS:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, np.transpose(mb_values,(0,2,1)) , mb_neglogpacs, mb_infos, mb_values_i, mb_returns_i, mb_states_nce, mb_anchors_nce, mb_labels_nce, np.transpose(mb_actions_nce,(0,2,3,1)) , mb_neglogps_nce, mb_rewards_nce, mb_infos_nce)),
                epinfos, eval_epinfos)
        else:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_infos, mb_rewards)),
             epinfos, eval_epinfos)

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


def learn(*, policy, env, eval_env, nsteps, total_timesteps, ent_coef, lr,
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

    runner = Runner(env=env, eval_env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    eval_epinfobuf100 = deque(maxlen=100)
    tfirststart = time.time()
    active_ep_buf = epinfobuf100
    eval_active_ep_buf = eval_epinfobuf100

    nupdates = total_timesteps//nbatch
    mean_rewards = []
    datapoints = []

    run_t_total = 0
    train_t_total = 0

    can_save = False
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

    z_iter = 0
    curr_z = np.random.randint(0, high=Config.POLICY_NHEADS)
    tb_writer = TB_Writer(sess)
    import os
    os.environ["WANDB_API_KEY"] = "02e3820b69de1b1fcc645edcfc3dd5c5079839a1"
    group_name = "%s__%s__%d__%d__%f__%d" %(Config.ENVIRONMENT,Config.RUN_ID,Config.CLUSTER_T,Config.N_KNN, Config.TEMP, Config.N_SKILLS)
    name = "%s__%s__%d__%d__%f__%d__%d" %(Config.ENVIRONMENT,Config.RUN_ID,Config.CLUSTER_T,Config.N_KNN,  Config.TEMP, Config.N_SKILLS, np.random.randint(100000000))
    wandb.init(project='ising_generalization' if Config.ENVIRONMENT == 'ising' else 'procgen_generalization' , entity='ssl_rl', config=Config.args_dict, group=group_name, name=name, mode="disabled" if Config.DISABLE_WANDB else "online")
    for update in range(start_update+1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        # if Config.CUSTOM_REP_LOSS:
        #     params = tf.compat.v1.trainable_variables()
        #     source_params = [p for p in params if p.name in model.train_model.RL_enc_param_names]
        #     for i in range(1,Config.POLICY_NHEADS):
        #         target_i_params = [p for p in params if p.name in model.train_model.target_enc_param_names[i]]
        #         soft_update(source_params,target_i_params,tau=0.95)

        mpi_print('collecting rollouts...')
        run_tstart = time.time()
        # if z_iter < 4: # 8 epochs / skill
        #     z_iter += 1
        # else:
        #     # sample new skill for current episodes
        #     curr_z = np.random.randint(0, high=Config.POLICY_NHEADS)
        #     model.head_idx_current_batch = curr_z
        #     z_iter = 0

        packed = runner.run(update_frac=update/nupdates)
    
        obs, returns, masks, actions, values, neglogpacs, infos, rewards, epinfos, eval_epinfos = packed
        values_i = returns_i = states_nce = anchors_nce = labels_nce = actions_nce = neglogps_nce = rewards_nce = infos_nce = None
    
        # reshape our augmented state vectors to match first dim of observation array
        # (mb_size*num_envs, 64*64*RGB)
        # (mb_size*num_envs, num_actions)
        avg_value = np.mean(values)
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)
        eval_epinfobuf100.extend(eval_epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed
        mpi_print('rollouts complete')

        mblossvals = []

        mpi_print('updating parameters...')
        train_tstart = time.time()

        mean_cust_loss = 0
        inds = np.arange(nbatch)
        inds_nce = np.arange(nbatch//runner.nce_update_freq)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            np.random.shuffle(inds_nce)
            for start in range(0, nbatch, nbatch_train):
                sess.run([model.train_model.train_dropout_assign_ops])
                end = start + nbatch_train
                mbinds = inds[start:end]

                
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, infos, values, neglogpacs, rewards))
                
                mblossvals.append(model.train(lrnow, cliprangenow, *slices, train_target='policy'))
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, infos, values, neglogpacs, rewards))
                model.train(lrnow, cliprangenow, *slices, train_target='encoder')
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, infos, values, neglogpacs, rewards))
                model.train(lrnow, cliprangenow, *slices, train_target='latent')
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
            eval_rew_mean = utils.process_ep_buf(eval_active_ep_buf, tb_writer=tb_writer, suffix='_eval', step=step)
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

            # eval_rew_mean = episode_rollouts(eval_env,model,step,tb_writer)

            mpi_print('eplenmean', ep_len_mean)
            mpi_print('eprew', rew_mean_10)
            mpi_print('eprew_eval', eval_rew_mean)
            mpi_print('fps', fps)
            mpi_print('total_timesteps', update*nbatch)
            mpi_print([epinfo['r'] for epinfo in epinfobuf10])

            rep_loss = 0
            if len(mblossvals):
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    mpi_print(lossname, lossval)
                    tb_writer.log_scalar(lossval, lossname, step=step)
            mpi_print('----\n')

            wandb.log({"%s/eprew"%(Config.ENVIRONMENT):rew_mean_10,
                        "%s/eprew_eval"%(Config.ENVIRONMENT):eval_rew_mean,
                        "%s/custom_step"%(Config.ENVIRONMENT):step})
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
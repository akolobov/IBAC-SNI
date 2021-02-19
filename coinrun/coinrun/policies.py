import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype, _matching_fc
from baselines.common.input import observation_input
from coinrun.ppo2 import MpiAdamOptimizer
# TODO this is no longer supported in tfv2, so we'll need to
# properly refactor where it's used if we want to use
# some of the options (e.g. beta)
#ds = tf.contrib.distributions
from mpi4py import MPI
from gym.spaces import Discrete, Box
from coinrun.config import Config


from coinrun.config import Config

def impala_cnn(images, depths=[16, 32, 32]):
    use_batch_norm = Config.USE_BATCH_NORM == 1
    slow_dropout_assign_ops = []
    fast_dropout_assign_ops = []

    def dropout_openai(out, rate, name):
        out_shape = out.get_shape().as_list()
        var_name = 'mask_{}'.format(name)
        batch_seed_shape = out_shape[1:]
        batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.compat.v1.random_uniform_initializer(minval=0, maxval=1), trainable=False)
        batch_seed_assign = tf.compat.v1.assign(batch_seed, tf.random.uniform(batch_seed_shape, minval=0, maxval=1))
        dout_assign_ops = [batch_seed_assign]
        curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - rate))
        curr_mask = curr_mask * (1.0 / (1.0 - rate))
        out = out * curr_mask
        return out, dout_assign_ops

    def conv_layer(out, depth, i):
        with tf.compat.v1.variable_scope("conv{}".format(i)):
            out = tf.compat.v1.layers.conv2d(out, depth, 3, padding='same')
            if use_batch_norm:
                out = tf.keras.layers.BatchNormalization()(x)
        return out

    def residual_block(inputs, twos):
        depth = inputs.get_shape()[-1].value
        out = tf.nn.relu(inputs)
        out = conv_layer(out, depth, twos[0])
        out = tf.nn.relu(out)
        out = conv_layer(out, depth, twos[1])
        return out + inputs

    def conv_sequence(inputs, depth, offsets):
        out = conv_layer(inputs, depth, offsets[0])
        out = tf.compat.v1.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out, offsets[1:3])
        out = residual_block(out, offsets[3:5])
        return out

    out = images
    for nr, depth in enumerate(depths):
        offsets = [x + 5*nr for x in range(5)]
        out = conv_sequence(out, depth, offsets)
    out = tf.compat.v1.layers.flatten(out)
    out = tf.nn.relu(out)

    # if Config.BETA >= 0:
    #     print("Creating VIB layer")
    #     params = tf.layers.dense(out, 256*2)
    #     mu, rho = params[:, :256], params[:, 256:]
    #     encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)

    #     with tf.compat.v1.variable_scope("info_loss"):
    #         prior = ds.Normal(0.0, 1.0)
    #         info_loss = tf.reduce_sum(tf.reduce_mean(
    #             ds.kl_divergence(encoding, prior), 0)) / np.log(2)

    #         # info_loss = tf.identity(info_loss, name="info_loss")
    #         tf.add_to_collection("INFO_LOSS", info_loss)
    #         # info_loss = tf.Print(info_loss, [info_loss])

    #     with tf.control_dependencies([info_loss]):
    #         batch_size = tf.shape(out)[0]
    #         # batch_size = tf.Print(batch_size, [tf.shape(out)])
    #         out = tf.reshape(
    #                 encoding.sample(Config.NR_SAMPLES),
    #                 shape=(batch_size * Config.NR_SAMPLES, 256))
    #         out_mean = mu

    # elif Config.BETA_L2A >= 0:
    #     print("Creating L2A regularized layer")
    #     out = tf.layers.dense(out, 256)
    #     with tf.compat.v1.variable_scope("info_loss"):
    #         info_loss = tf.reduce_sum(tf.reduce_mean(tf.square(out), 0))
    #     tf.add_to_collection("INFO_LOSS_L2A", info_loss)

    #     with tf.control_dependencies([info_loss]):
    #         out = tf.identity(out)
    #     out_mean = out
    # elif Config.DROPOUT > 0:
    #     print("Creating Dropout layer")
    #     out_mean = tf.layers.dense(out, 256)
    #     out = tf.nn.dropout(out_mean, rate=Config.DROPOUT)
    # elif Config.DROPOUT > 0:
    #     print("Creating Dropout layer")
    #     latent = tf.layers.dense(out, 256)
    #     out, fast_dropout_assign_ops = dropout_openai(latent, rate=Config.DROPOUT, name='fast')
    #     out_mean, slow_dropout_assign_ops = dropout_openai(latent, rate=Config.DROPOUT, name='slow')
    # else:
    core = out
    with tf.compat.v1.variable_scope("dense0"):
        act_invariant = tf.compat.v1.layers.dense(core, Config.NODES)
        act_invariant = tf.identity(act_invariant, name="action_invariant_layers")
        act_invariant = tf.nn.relu(act_invariant)
    with tf.compat.v1.variable_scope("dense1"):
        act_condit = tf.compat.v1.layers.dense(core, 256 - Config.NODES)
        act_condit = tf.identity(act_condit, name="action_conditioned_layers")
        act_condit = tf.nn.relu(act_condit)
    return act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assign_ops

def choose_cnn(images):
    arch = Config.ARCHITECTURE
    scaled_images = tf.cast(images, tf.float32) / 255.

    if arch == 'nature':
        raise NotImplementedError()
        out = nature_cnn(scaled_images)
    elif arch == 'impala':
        return impala_cnn(scaled_images)
    elif arch == 'impalalarge':
        return impala_cnn(scaled_images, depths=[32, 64, 64, 64, 64])
    else:
        assert(False)


class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, max_grad_norm, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        self.rep_loss = None
        # explicitly create  vector space for latent vectors
        latent_space = Box(-np.inf, np.inf, shape=(256,))
        # So that I can compute the saliency map
        if Config.REPLAY:
            X = tf.compat.v1.placeholder(shape=(nbatch,) + ob_space.shape, dtype=np.float32, name='Ob')
            processed_x = X
        else:
            X, processed_x = observation_input(ob_space, nbatch)
            REP_PROC = tf.compat.v1.placeholder(dtype=tf.float32, shape=(nbatch, 64, 64, 3))
            Z = tf.compat.v1.placeholder(dtype=tf.float32, shape=(nbatch, Config.N_SKILLS), name='Skill')
            # trajectories of length m, for N policy heads.
            self.STATE = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
            self.STATE_NCE = tf.compat.v1.placeholder(tf.float32, [Config.REP_LOSS_M,Config.POLICY_NHEADS,Config.NUM_ENVS,64,64,3])
            self.ANCH_NCE = tf.compat.v1.placeholder(tf.float32, [Config.NUM_ENVS,64,64,3])
            # labels of Q value quantile bins
            self.LAB_NCE = tf.compat.v1.placeholder(tf.float32, [Config.POLICY_NHEADS,Config.NUM_ENVS])
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(processed_x)
            self.train_dropout_assign_ops = fast_dropout_assigned_ops
            self.run_dropout_assign_ops = slow_dropout_assign_ops
            # stack together action invariant & conditioned layers for full representation layer
            self.h =  tf.concat([act_condit, act_invariant], axis=1)

            # NOTE: (Ahmed) I commented out all the IBAC-SNI settings to make this easier to read
            # since we shouldn't be using any of these settings anyway.
            # Noisy policy and value function for train
            # if Config.BETA >= 0:
            #     pdparam = _matching_fc(self.h, 'pi', ac_space.n, init_scale=1.0, init_bias=0)
            #     pdparam = tf.reshape(pdparam, shape=(Config.NR_SAMPLES, -1, ac_space.n))
            #     pdparam = tf.transpose(pdparam, perm=[1,0,2])

            #     dists = ds.Categorical(logits=pdparam)
            #     self.pd_train = ds.MixtureSameFamily(
            #         mixture_distribution=ds.Categorical(probs=[1./Config.NR_SAMPLES]*Config.NR_SAMPLES),
            #         components_distribution=dists)
            #     self.pd_train.neglogp = lambda a: - self.pd_train.log_prob(a)
            #     self.vf_train = tf.reduce_mean(tf.reshape(fc(self.h, 'v', 1), shape=(Config.NR_SAMPLES, -1, 1)), 0)[:, 0]
            # else:

        # create phi(s') using the same encoder
        with tf.variable_scope("model", reuse=True) as scope:
            first, second, _, _ = choose_cnn(tf.reshape(self.STATE_NCE,(-1,64,64,3)))
            # ( m * K * N, hidden_dim)
            self.phi_traj_nce = tf.concat([first, second], axis=1)

            first, second, _, _ = choose_cnn(self.ANCH_NCE)
            self.phi_anch_nce = tf.concat([first, second], axis=1)

            first, second, _, _ = choose_cnn(self.STATE)
            self.phi_STATE = tf.concat([first, second], axis=1)

        
        with tf.compat.v1.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
                self.pd_train = []
                for i in range(Config.POLICY_NHEADS):
                    with tf.compat.v1.variable_scope("head_"+str(i), reuse=tf.compat.v1.AUTO_REUSE):
                        self.pd_train.append(self.pdtype.pdfromlatent(self.h, init_scale=0.01)[0])
            elif Config.AGENT == 'ppo_diayn':
                with tf.compat.v1.variable_scope("head_0", reuse=tf.compat.v1.AUTO_REUSE):
                    self.h = tf.concat([self.h, Z], axis=1)
                    self.pd_train = [self.pdtype.pdfromlatent(self.h, init_scale=0.01)[0]]
            else:
                with tf.compat.v1.variable_scope("head_0", reuse=tf.compat.v1.AUTO_REUSE):
                    self.pd_train = [self.pdtype.pdfromlatent(self.h, init_scale=0.01)[0]]
            
            if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
                self.vf_train = [fc(self.h, 'v'+str(i), 1)[:, 0] for i in range(Config.POLICY_NHEADS)]
            else:
                self.vf_train = [fc(self.h, 'v_0', 1)[:, 0] ]
            self.vf_i_train = fc(self.h, 'v_i', 1)[:, 0] # PPO+RND only

            # Plain Dropout version: Only fast updates / stochastic latent for VIB
            self.pd_run = self.pd_train
            self.vf_run = self.vf_train
            self.vf_i_run = self.vf_i_train

            # For Dropout: Always change layer, so slow layer is never used
            self.run_dropout_assign_ops = []

        # Use the current head for classical PPO updates
        a0_run = [self.pd_run[head_idx].sample() for head_idx in range(Config.POLICY_NHEADS)]
        neglogp0_run = [self.pd_run[head_idx].neglogp(a0_run[head_idx]) for head_idx in range(Config.POLICY_NHEADS)]
        self.initial_state = None

        def step(ob, update_frac, head_idx, one_hot_skill=None, *_args, **_kwargs):
            if Config.REPLAY:
                ob = ob.astype(np.float32)
            if Config.AGENT == 'ppo_rnd':
                a, v,v_i, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], self.vf_i_run, neglogp0_run[head_idx]], {X: ob})
                return a, v, v_i, self.initial_state, neglogp
            elif  Config.AGENT == 'ppo_diayn':
                a, v,v_i, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], self.vf_i_run, neglogp0_run[head_idx]], {X: ob, Z: one_hot_skill})
                return a, v, v_i, self.initial_state, neglogp
            elif not Config.CUSTOM_REP_LOSS:
                head_idx = 0
                a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob})
                return a, v, self.initial_state, neglogp
            else:
                # a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run, neglogp0_run[head_idx]], {X: ob})
                rets = sess.run(a0_run + self.vf_run + neglogp0_run, {X: ob})
                a = rets[:Config.POLICY_NHEADS]
                v = rets[Config.POLICY_NHEADS:2*Config.POLICY_NHEADS]
                neglogp = rets[2*Config.POLICY_NHEADS:]
                return a, v, self.initial_state, neglogp

        def rep_vec(ob, *_args, **_kwargs):
            return sess.run(self.h, {X: ob})

        def value(ob, update_frac, *_args, **_kwargs):
            return sess.run(self.vf_run, {X: ob})

        def value_i(ob, update_frac, *_args, **_kwargs): 
            return sess.run(self.vf_i_run, {X: ob})

        def custom_train(ob, rep_vecs):
            return sess.run([self.rep_loss], {X: ob, REP_PROC: rep_vecs})[0]

        self.X = X
        self.processed_x = processed_x
        self.step = step
        self.value = value
        self.value_i = value_i
        self.rep_vec = rep_vec
        self.custom_train = custom_train
        self.encoder = choose_cnn
        self.REP_PROC = REP_PROC


def get_policy():
    use_lstm = Config.USE_LSTM
    
    if use_lstm == 1:
        raise NotImplementedError()
        policy = LstmPolicy
    elif use_lstm == 0:
        policy = CnnPolicy
    else:
        assert(False)

    return policy
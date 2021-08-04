import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype, _matching_fc
from baselines.common.input import observation_input
from coinrun.ppo2_goal import sinkhorn
from coinrun.models import FiLM, TemporalBlock
# TODO this is no longer supported in tfv2, so we'll need to
# properly refactor where it's used if we want to use
# some of the options (e.g. beta)
#ds = tf.contrib.distributions
from mpi4py import MPI
from gym.spaces import Discrete, Box
from coinrun.config import Config

from tensorflow.keras import initializers


from coinrun.config import Config

def impala_cnn(images, depths=[16, 32, 32], prefix=""):
    use_batch_norm = Config.USE_BATCH_NORM == 1
    slow_dropout_assign_ops = []
    fast_dropout_assign_ops = []

    def dropout_openai(out, rate, name):
        out_shape = out.get_shape().as_list()
        var_name = prefix+'mask_{}'.format(name)
        batch_seed_shape = out_shape[1:]
        batch_seed = tf.compat.v1.get_variable(var_name, shape=batch_seed_shape, initializer=tf.compat.v1.random_uniform_initializer(minval=0, maxval=1), trainable=False)
        batch_seed_assign = tf.compat.v1.assign(batch_seed, tf.random.uniform(batch_seed_shape, minval=0, maxval=1))
        dout_assign_ops = [batch_seed_assign]
        curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - rate))
        curr_mask = curr_mask * (1.0 / (1.0 - rate))
        out = out * curr_mask
        return out, dout_assign_ops

    def conv_layer(out, depth, i):
        with tf.compat.v1.variable_scope("{}conv{}".format(prefix,i)):
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
    core = out
    with tf.compat.v1.variable_scope(prefix+"dense0"):
        act_invariant = tf.compat.v1.layers.dense(core, Config.NODES)
        act_invariant = tf.identity(act_invariant, name="action_invariant_layers")
        act_invariant = tf.nn.relu(act_invariant)
    with tf.compat.v1.variable_scope(prefix+"dense1"):
        act_condit = tf.compat.v1.layers.dense(core, 256 - Config.NODES)
        act_condit = tf.identity(act_condit, name="action_conditioned_layers")
        act_condit = tf.nn.relu(act_condit)
    return act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assign_ops

def choose_cnn(images,prefix=""):
    arch = Config.ARCHITECTURE
    scaled_images = tf.cast(images, tf.float32) / 255.

    if arch == 'nature':
        raise NotImplementedError()
        out = nature_cnn(scaled_images)
    elif arch == 'impala':
        return impala_cnn(scaled_images,prefix=prefix)
    elif arch == 'impalalarge':
        return impala_cnn(scaled_images, depths=[32, 64, 64, 64, 64], prefix=prefix)
    else:
        assert(False)

def get_rnd_predictor(trainable):
    inputs = tf.keras.layers.Input((256, ))
    p = tf.keras.layers.Dense(1024,trainable=trainable)(inputs)
    h = tf.keras.Model(inputs, p)
    return h

def get_latent_discriminator():
    inputs = tf.keras.layers.Input((256, ))
    p = tf.keras.layers.Dense(512,activation='relu')(inputs)
    p2 = tf.keras.layers.Dense(Config.N_SKILLS)(p)
    h = tf.keras.Model(inputs, p2)
    return h

def get_seq_encoder():
    inputs = tf.keras.layers.Input((Config.REP_LOSS_M,320 ))
    conv1x1 = tf.keras.layers.Conv1D(filters=512,kernel_size=(1),activation='relu')(inputs)
    p = tf.keras.layers.Dense(256,activation='relu')(tf.reshape(conv1x1,(-1,Config.REP_LOSS_M*512)))
    # output should be (ne, N, x)
    # p = tf.reshape(p,(-1,256))
    h = tf.keras.Model(inputs, p)
    return h

def get_action_encoder(n_actions):
    inputs = tf.keras.layers.Input((n_actions ))
    p = tf.keras.layers.Dense(64,activation='relu')(inputs)
    h = tf.keras.Model(inputs, p)
    return h

def get_anch_encoder():
    inputs = tf.keras.layers.Input((256 ))
    p = tf.keras.layers.Dense(256,activation='relu')(inputs)
    h = tf.keras.Model(inputs, p)
    return h

def get_predictor(n_in=256,n_out=256,prefix='predictor'):
    inputs = tf.keras.layers.Input((n_in, ))
    p = tf.keras.layers.Dense(256,activation='relu',name=prefix+'_hidden1')(inputs)
    p2 = tf.keras.layers.Dense(n_out,name=prefix+'_hidden2')(p)
    h = tf.keras.Model(inputs, p2)
    return h

def get_linear_layer(n_in=256,n_out=128,prefix='linear_predictor',init=None):
    inputs = tf.keras.layers.Input((n_in, ))
    if init is None:
        p = tf.keras.layers.Dense(n_out,name=prefix+'_linear')(inputs)
    else:
        p = tf.keras.layers.Dense(n_out,name=prefix+'_linear',kernel_initializer=init,bias_initializer=initializers.Zeros())(inputs)
        
    h = tf.keras.Model(inputs, p)
    return h

def get_online_predictor(n_in=128,n_out=128,prefix='online_predictor'):
    inputs = tf.keras.layers.Input((n_in,))
    p = tf.keras.layers.Dense(128, activation='relu',name=prefix+'_hidden1')(inputs)
    p2 = tf.keras.layers.Dense(512, activation='relu',name=prefix+'_hidden2')(p)
    p3 = tf.keras.layers.Dense(n_out,name=prefix+'_hidden3')(p2)
    h = tf.keras.Model(inputs, p3)
    return h

def get_transition_model(n_in=256+15,n_out=256,prefix='transition_model'):
    inputs = tf.keras.layers.Input((n_in,))
    p = tf.keras.layers.Dense(256, activation=None,name=prefix+'_hidden1')(inputs)
    p = tf.keras.layers.LayerNormalization()(p)
    p = tf.keras.layers.ReLU()(p)
    p = tf.keras.layers.Dense(256)(p)
    h = tf.keras.Model(inputs, p)
    return h

def get_time_conv():
    def h(x):
        # T=256
        # x = tf.layers.Conv1D(128,32,8, activation='relu')(x)
        # x = tf.layers.Conv1D(256,16,2, activation='relu')(x)
        # x = tf.layers.Conv1D(128,6,2, activation='relu')(x)
        x = tf.layers.Conv1D(128,4,2, activation='relu')(x)
        x = tf.layers.Conv1D(128,3,2, activation=None)(x)
        return x
    return h

def tanh_clip(x, clip_val=20.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    Trick from AM-DIM
    '''
    if clip_val is not None:
        # why not just clip_val * tanh(x), since tanh : R -> [-1, 1]
        x_clip = clip_val * tf.math.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip

def cos_loss(p, z):
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    dist = 2-2*tf.reduce_sum(input_tensor=(p*z), axis=1)
    return dist

def _compute_distance(x, y):
    y = tf.stop_gradient(y)
    x = tf.math.l2_normalize(x, axis=1)
    y = tf.math.l2_normalize(y, axis=1)

    dist = 2 - 2 * tf.reduce_sum(tf.reshape(x,(-1, 1, x.shape[1])) *
                                tf.reshape(y,(1, -1, y.shape[1])), -1)
    return dist

def shuffle_custom(x):
    return tf.gather(x, tf.random.shuffle(tf.range(tf.shape(x)[0])))
"""
PSE helper methods
"""
def _calculate_action_cost_matrix(ac1, ac2):
  diff = tf.expand_dims(ac1, axis=1) - tf.expand_dims(ac2, axis=0)
  return tf.cast(tf.reduce_mean(tf.abs(diff), axis=-1), dtype=tf.float32)

def metric_fixed_point_fast(cost_matrix, gamma=0.99, eps=1e-7):
  """Dynamic prograaming for calculating PSM."""
  d = np.zeros_like(cost_matrix)
  def operator(d_cur):
    d_new = 1 * cost_matrix
    discounted_d_cur = gamma * d_cur
    d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
    d_new[:-1, -1] += discounted_d_cur[1:, -1]
    d_new[-1, :-1] += discounted_d_cur[-1, 1:]
    return d_new

  while True:
    d_new = operator(d)
    if np.sum(np.abs(d - d_new)) < eps:
      break
    else:
      d = d_new[:]
  return d

def compute_psm_metric(actions1, actions2, gamma):
  action_cost = _calculate_action_cost_matrix(actions1, actions2)
  return tf_metric_fixed_point(action_cost, gamma=gamma)

def tf_metric_fixed_point(action_cost_matrix, gamma):
  return tf.numpy_function(
      metric_fixed_point_fast, [action_cost_matrix, gamma], Tout=tf.float32)

EPS = 1e-9
def cosine_similarity(x, y):
  """Computes cosine similarity between all pairs of vectors in x and y."""
  x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
  similarity_matrix = tf.reduce_sum(x_expanded * y_expanded, axis=-1)
  similarity_matrix /= (
      tf.norm(x_expanded, axis=-1) * tf.norm(y_expanded, axis=-1) + EPS)
  return similarity_matrix

def sample_indices(dim_x, size=128, sort=False):
  dim_x = tf.cast(dim_x, tf.int32)
  indices = tf.range(0, dim_x, dtype=tf.int32)
  indices = tf.random.shuffle(indices)[:size]
  if sort:
    indices = tf.sort(indices)
  return indices

def representation_alignment_loss(representation_1,
                                  representation_2,
                                  metric_vals,
                                  use_coupling_weights=False,
                                  coupling_temperature=0.1,
                                  return_representation=False,
                                  temperature=1.0):
  """PSE loss."""
  if np.random.randint(2) == 1:
    # obs2, obs1 = obs1, obs2
    representation_1, representation_2 = representation_2, representation_1
    metric_vals = tf.transpose(metric_vals)

  indices = sample_indices(tf.shape(metric_vals)[0], sort=return_representation)
#   obs1 = tf.gather(obs1, indices, axis=0)
  metric_vals = tf.gather(metric_vals, indices, axis=0)

  similarity_matrix = cosine_similarity(representation_1, representation_2)
  alignment_loss = contrastive_loss(
      similarity_matrix,
      metric_vals,
      temperature,
      coupling_temperature=coupling_temperature,
      use_coupling_weights=use_coupling_weights)

  if return_representation:
    return alignment_loss, similarity_matrix
  else:
    return alignment_loss

def contrastive_loss(similarity_matrix,
                     metric_values,
                     temperature,
                     coupling_temperature=1.0,
                     use_coupling_weights=True):
  """Contrative Loss with soft coupling."""
  print('Using alternative contrastive loss.')
  metric_shape = tf.shape(metric_values)
  similarity_matrix /= temperature
  neg_logits1 = similarity_matrix

  col_indices = tf.cast(tf.argmin(metric_values, axis=1), dtype=tf.int32)
  pos_indices1 = tf.stack(
      (tf.range(metric_shape[0], dtype=tf.int32), col_indices), axis=1)
  pos_logits1 = tf.gather_nd(similarity_matrix, pos_indices1)
  
  if use_coupling_weights:
    metric_values /= coupling_temperature
    coupling = tf.exp(-metric_values)
    pos_weights1 = -tf.gather_nd(metric_values, pos_indices1)
    pos_logits1 += pos_weights1
    negative_weights = tf.math.log((1.0 - coupling) + EPS)
    neg_logits1 += tf.tensor_scatter_nd_update(negative_weights, pos_indices1,pos_weights1)
  neg_logits1 = tf.math.reduce_logsumexp(neg_logits1, axis=1)
  return tf.reduce_mean(neg_logits1 - pos_logits1)

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
            X, processed_x = observation_input(ob_space, None)
            TRAIN_NUM_STEPS = Config.NUM_STEPS//16
            REP_PROC = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3), name='Rep_Proc')
            Z_INT = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name='Curr_Skill_idx')
            Z = tf.compat.v1.placeholder(dtype=tf.float32, shape=(nbatch, Config.N_SKILLS), name='Curr_skill')
            CODES = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1024, Config.N_SKILLS), name='Train_Codes')
            CLUSTER_DIMS = 256
            HIDDEN_DIMS_SSL = 256
            STEP_BOOL = tf.placeholder(tf.bool, shape=[])
            self.protos = tf.compat.v1.Variable(initial_value=tf.random.normal(shape=(CLUSTER_DIMS, Config.N_SKILLS)), trainable=True, name='Prototypes')
            self.A = self.pdtype.sample_placeholder([None],name='A')
            self.R = tf.compat.v1.placeholder(tf.float32, [None], name='R')
            # trajectories of length m, for N policy heads.
            self.STATE = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
            self.STATE_NCE = tf.compat.v1.placeholder(tf.float32, [Config.REP_LOSS_M,1,None,64,64,3])
            self.ANCH_NCE = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
            # labels of Q value quantile bins
            self.LAB_NCE = tf.compat.v1.placeholder(tf.float32, [Config.POLICY_NHEADS,None])
            self.A_i = self.pdtype.sample_placeholder([None,Config.REP_LOSS_M,1],name='A_i')
            self.R_cluster = tf.compat.v1.placeholder(tf.float32, [None])
            self.A_cluster = self.pdtype.sample_placeholder([None, Config.NUM_ENVS], name='A_cluster')

            self.pse_obs_1 = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
            self.pse_actions_1 = self.pdtype.sample_placeholder([None],name='A_1')
            self.pse_rewards_1 = tf.compat.v1.placeholder(tf.float32, [None], name='R_1')
            self.pse_obs_2 = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
            self.pse_actions_2 = self.pdtype.sample_placeholder([None],name='A_2')
            self.pse_rewards_2 = tf.compat.v1.placeholder(tf.float32, [None], name='R_2')
            
        
        with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
            act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(processed_x)
            self.train_dropout_assign_ops = fast_dropout_assigned_ops
            self.run_dropout_assign_ops = slow_dropout_assign_ops
            self.h =  tf.concat([act_condit, act_invariant], axis=1)

        """
        PSEs code
        """
        contrastive_loss_temperature = 0.5
        with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
            n_pse = tf.shape(self.pse_obs_1)[0]
            concat_pse_obs = tf.concat([self.pse_obs_1,self.pse_obs_2],0)
            act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(concat_pse_obs)
            h_pse =  tf.concat([act_condit, act_invariant], axis=1)
            representation_1, representation_2 = h_pse[:n_pse], h_pse[n_pse:]
            # PSE loss
            metric_vals = compute_psm_metric(tf.one_hot(self.pse_actions_1,15),tf.one_hot(self.pse_actions_2,15),Config.GAMMA)
            self.contrastive_loss = representation_alignment_loss(
                        representation_1,
                        representation_2,
                        metric_vals,
                        use_coupling_weights=False,
                        temperature=contrastive_loss_temperature,
                        return_representation=False)
    

        with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
            
            with tf.compat.v1.variable_scope("head_0", reuse=tf.compat.v1.AUTO_REUSE):
                self.pd_train = [self.pdtype.pdfromlatent(tf.stop_gradient(self.h), init_scale=0.01)[0]]
            
            self.vf_train = [fc(self.h, 'v_0', 1)[:, 0] ]

            # Plain Dropout version: Only fast updates / stochastic latent for VIB
            self.pd_run = self.pd_train
            self.vf_run = self.vf_train
            

            # For Dropout: Always change layer, so slow layer is never used
            self.run_dropout_assign_ops = []


        # Use the current head for classical PPO updates
        a0_run = [self.pd_run[head_idx].sample() for head_idx in range(Config.POLICY_NHEADS)]
        neglogp0_run = [self.pd_run[head_idx].neglogp(a0_run[head_idx]) for head_idx in range(Config.POLICY_NHEADS)]
        self.initial_state = None

        def step(ob, update_frac, skill_idx=None, one_hot_skill=None, nce_dict = {},  *_args, **_kwargs):
            if Config.REPLAY:
                ob = ob.astype(np.float32)
            
            head_idx = 0
            a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob})
            return a, v, self.initial_state, neglogp
            

        def rep_vec(ob, *_args, **_kwargs):
            return sess.run(self.h, {X: ob})

        def value(ob, update_frac, one_hot_skill=None, *_args, **_kwargs):
            if Config.AGENT == 'ppo_diayn':
                return sess.run(self.vf_run, {X: ob, Z: one_hot_skill})
            elif Config.AGENT == 'ppo_goal':
                return sess.run(self.vf_run, {REP_PROC: ob, Z: one_hot_skill})
            else:
                 return sess.run(self.vf_run, {self.STATE: ob, X:ob})


        def value_i(ob, update_frac, one_hot_skill=None, *_args, **_kwargs):
            if Config.AGENT == 'ppo_diayn':
                return sess.run(self.vf_i_run, {X: ob, Z: one_hot_skill})
            elif Config.AGENT == 'ppo_goal':
                return sess.run(self.vf_i_run, {REP_PROC: ob, Z: one_hot_skill})
            else:
                 return sess.run(self.vf_i_run, {self.STATE: ob, X:ob})

        def nce_fw_pass(nce_dict):
            return sess.run([self.vf_i_run,self.rep_loss],nce_dict)

        def custom_train(ob, rep_vecs):
            return sess.run([self.rep_loss], {X: ob, REP_PROC: rep_vecs})[0]
        
        def compute_codes(ob,act):
            return sess.run([tf.reshape(self.codes , (Config.NUM_ENVS,Config.NUM_STEPS,-1)), tf.reshape(self.u_t , (Config.NUM_ENVS,Config.NUM_STEPS,-1)), tf.reshape(self.z_t_1 , (Config.NUM_ENVS,Config.NUM_STEPS,-1)) , self.h_codes[:,1:]], {REP_PROC: ob, self.A_cluster: act})
        
        def compute_hard_codes(ob):
            return sess.run([self.codes, self.u_t, self.z_t_1], {REP_PROC: ob})

        def compute_cluster_returns(returns):
            return sess.run([self.cluster_returns],{self.R_cluster:returns})

        self.X = X
        self.processed_x = processed_x
        self.step = step
        self.value = value
        self.value_i = value_i
        self.rep_vec = rep_vec
        self.custom_train = custom_train
        self.nce_fw_pass = nce_fw_pass
        self.encoder = choose_cnn
        self.REP_PROC = REP_PROC
        self.Z = Z
        self.compute_codes = compute_codes
        self.compute_hard_codes = compute_hard_codes
        self.compute_cluster_returns = compute_cluster_returns
        self.CODES = CODES
        self.STEP_BOOL = STEP_BOOL


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
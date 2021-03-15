import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype, _matching_fc
from baselines.common.input import observation_input
from coinrun.ppo2_goal import sinkhorn
# TODO this is no longer supported in tfv2, so we'll need to
# properly refactor where it's used if we want to use
# some of the options (e.g. beta)
#ds = tf.contrib.distributions
from mpi4py import MPI
from gym.spaces import Discrete, Box
from coinrun.config import Config


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

def get_predictor(n_in=256,n_out=256):
    inputs = tf.keras.layers.Input((n_in, ))
    p = tf.keras.layers.Dense(256,activation='relu')(inputs)
    p2 = tf.keras.layers.Dense(n_out)(p)
    h = tf.keras.Model(inputs, p2)
    return h

def get_linear_layer(n_in=256,n_out=128):
    inputs = tf.keras.layers.Input((n_in, ))
    p = tf.keras.layers.Dense(n_out)(inputs)
    h = tf.keras.Model(inputs, p)
    return h

def get_online_predictor():
    inputs = tf.keras.layers.Input((128,))
    p = tf.keras.layers.Dense(128, activation='relu')(inputs)
    p2 = tf.keras.layers.Dense(512, activation='relu')(p)
    p3 = tf.keras.layers.Dense(128)(p2)
    h = tf.keras.Model(inputs, p3)
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
    return -tf.reduce_sum(input_tensor=(p*z), axis=1)

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
            REP_PROC = tf.compat.v1.placeholder(dtype=tf.float32, shape=(32, 32, 64, 64, 3), name='Rep_Proc')
            Z_INT = tf.compat.v1.placeholder(dtype=tf.int32, shape=(), name='Curr_Skill_idx')
            Z = tf.compat.v1.placeholder(dtype=tf.float32, shape=(nbatch, Config.N_SKILLS), name='Curr_skill')
            self.protos = tf.compat.v1.Variable(initial_value=tf.random.normal(shape=(128, Config.N_SKILLS)), trainable=True, name='Prototypes')
            self.A = self.pdtype.sample_placeholder([None],name='A')
            # trajectories of length m, for N policy heads.
            self.STATE = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
            self.STATE_NCE = tf.compat.v1.placeholder(tf.float32, [Config.REP_LOSS_M,1,None,64,64,3])
            self.ANCH_NCE = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
            # labels of Q value quantile bins
            self.LAB_NCE = tf.compat.v1.placeholder(tf.float32, [Config.POLICY_NHEADS,None])
            self.A_i = self.pdtype.sample_placeholder([None,Config.REP_LOSS_M,1],name='A_i')
        if Config.AGENT == 'ppo_goal':
             with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
                act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(processed_x)
                self.train_dropout_assign_ops = fast_dropout_assigned_ops
                self.run_dropout_assign_ops = slow_dropout_assign_ops
                self.h =  tf.concat([act_condit, act_invariant], axis=1)
        else:
            with tf.compat.v1.variable_scope("model_0", reuse=tf.compat.v1.AUTO_REUSE):
                act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(processed_x)
                self.train_dropout_assign_ops = fast_dropout_assigned_ops
                self.run_dropout_assign_ops = slow_dropout_assign_ops
                self.h =  tf.concat([act_condit, act_invariant], axis=1)
        if Config.AGENT == 'ppg':
            self.X_pi, self.processed_x_pi = observation_input(ob_space, None)
            with tf.compat.v1.variable_scope("pi_branch", reuse=tf.compat.v1.AUTO_REUSE):
                act_condit_pi, act_invariant_pi, _, _ = choose_cnn(processed_x,prefix='')
                self.h_pi =  tf.concat([act_condit_pi, act_invariant_pi], axis=1)
                act_one_hot = tf.reshape(tf.one_hot(self.A,ac_space.n), (-1,ac_space.n))
                self.adv_pi = get_predictor(n_in=256+15,n_out=1)(tf.concat([self.h_pi,act_one_hot],axis=1))
                self.v_pi = get_predictor(n_in=256,n_out=1)(self.h_pi)

        if Config.AGENT == 'ppo_diayn':
            # with tf.variable_scope("model", reuse=True) as scope:
            """
            DIAYN loss
            """
            self.discriminator = get_latent_discriminator()
            self.discriminator_logits = self.discriminator(tf.stop_gradient(self.h))
            self.discriminator_log_probs = tf.nn.log_softmax(self.discriminator_logits)
            
            self.skill_log_prob = tf.gather(self.discriminator_log_probs, Z_INT, axis=1)
            
            # condition on current skill
            self.h = tf.concat([self.h, Z], axis=1)

        elif Config.AGENT == 'ppo' and Config.CUSTOM_REP_LOSS and Config.REP_LOSS_WEIGHT > 0:
            # create phi(s') using the same encoder               
            with tf.variable_scope("model_0", reuse=tf.AUTO_REUSE) as scope:
                first, second, _, _ = choose_cnn(tf.reshape(self.STATE_NCE,(-1,64,64,3)))
                # ( m * K * N, hidden_dim)
                h = tf.concat([first, second], axis=1)
                latent_h = tf.transpose(tf.reshape( h ,(Config.REP_LOSS_M,-1,256)),perm=[1,0,2])
                act_one_hot = tf.reshape(tf.one_hot(self.A_i[:,:,0],ac_space.n), (-1,ac_space.n))
                phi_actions = tf.reshape(get_action_encoder(ac_space.n)( act_one_hot ), (-1,Config.REP_LOSS_M, 64) )
            params = tf.compat.v1.trainable_variables()
            self.RL_enc_param_names = [p.name for p in params if 'model_0/' in p.name]
            self.target_enc_param_names = []
            self.phi_traj_nce = []
            for i in range(0, Config.POLICY_NHEADS):
                with tf.variable_scope("model_%d"%(i), reuse=tf.AUTO_REUSE) as scope:
                    if i > 0:
                        h = tf.stop_gradient( latent_h )
                    else:
                        h = latent_h     
                    params = tf.compat.v1.trainable_variables()
                    self.target_enc_param_names.append([p.name for p in params if 'model_%d/'%i in p.name])
                    # concat actions with random state embeddings
                    # s_a_phi: n_batch x m x (n_rkhs_s + n_rkhs_a)
                    s_a_phi = tf.concat([h,phi_actions],2)
                    z_seq = get_seq_encoder()( s_a_phi )
                    self.phi_traj_nce.append( z_seq )
            
            with tf.variable_scope("model_0", reuse=True) as scope:

                first, second, _, _ = choose_cnn(self.ANCH_NCE)
                self.phi_anch_nce = tf.concat([first, second], axis=1)

                first, second, _, _ = choose_cnn(self.STATE)
                self.phi_STATE = tf.concat([first, second], axis=1)
                # m: length of NCE rollout (sub-traj), n_heads: number of heads, n_rkhs: latent dim (256 usually)

                self_sup_loss_type = 'BYOL' # infoNCE / BYOL 
                # (m, n_heads, n_batch, n_rkhs)
                
                # z_seq: n_batch x n_heads x n_rkhs. Global representation of z_{t+1:t+m}^k= f( phi(s_t+1),..,phi(s_t+m) ) for head k. Uses 1x1 Conv2D then MLP on flattened features of size m*256x256
                z_seq = tf.stack(self.phi_traj_nce)
                # z_anch: n_batch x n_rkhs. Global representation of z_t^k=h( phi(s_t) ) for head k by passing into 256x256 MLP "h".
                z_anch = get_anch_encoder()(self.phi_anch_nce)
                
                self.rep_loss = 0.
                if self_sup_loss_type == 'infoNCE':
                    # outer_prod: n_loc x n_batch x n_batch
                    # Use <z_t^k,z_{t+1:t+m}^k> as positives, 
                    #     <z_t^k,z_{t+1:t+m}^k'> for k!=k' as negatives
                    outer_prod = tanh_clip( tf.einsum("ijk,lk->jil",z_seq,z_anch) ) / (256)**0.5
                    for i in range(Config.POLICY_NHEADS):
                        # Mask out the labels for the respective heads. mask=1 if samples from same head, 0 otherwise
                        filter_ = tf.cast(tf.fill(tf.shape(self.LAB_NCE), i),tf.float32)
                        mask = tf.math.equal(filter_ , self.LAB_NCE)
                        mask_mul = tf.cast(tf.tile(tf.expand_dims(mask,-1),[1,1,tf.shape(z_anch)[0]]),tf.float32)
                        
                        # n_batch x n_heads
                        pos_scores = tf.transpose(tf.reduce_mean((mask_mul*outer_prod),2),(1,0))
                        # n_batch x n_batch x n_heads
                        neg_scores = tf.transpose(( (1.-mask_mul) * outer_prod) - (20 * mask_mul),(1,2,0))
                        shape_neg = tf.shape(neg_scores)
                        neg_scores = tf.reshape(neg_scores,(shape_neg[0],shape_neg[1]*shape_neg[2]))
                        mask_neg = tf.reshape((1.-mask_mul),(shape_neg[0],shape_neg[1]*shape_neg[2]))
                        neg_maxes = tf.reduce_max(neg_scores, 1, keepdims=True)
                        # n_batch x 1
                        neg_sumexp = tf.reduce_mean((mask_neg * tf.exp(neg_scores - neg_maxes)),1, keepdims=True)
                        # n_batch x n_heads
                        all_logsumexp = tf.log(tf.exp(pos_scores - neg_maxes) + neg_sumexp)
                        pos_shiftexp = pos_scores - neg_maxes
                        nce_scores = pos_shiftexp - all_logsumexp

                        self.rep_loss = self.rep_loss +  1/Config.POLICY_NHEADS * tf.reduce_mean(nce_scores,1)

                elif self_sup_loss_type == 'BYOL':
                    # z_anch: n_envs x n_heads x 512

                    p_seq_0 = get_predictor(n_out=256)(z_seq[0])
                    p_anch = get_predictor(n_out=256)(z_anch)
                    # z_anch = tf.tile(tf.reshape(z_anch,(Config.NUM_ENVS,1,-1)),tf.constant([1,Config.POLICY_NHEADS,1],tf.int32))
                    for i in range(1,Config.POLICY_NHEADS):
                        
                        # pred_Z = get_predictor(1)(tf.reshape(phi_traj_nce[:,i],(-1,256))) # count_latent_factors(Config.ENVIRONMENT)
                        # rep_loss += 1/Config.POLICY_NHEADS * tf.reduce_mean(input_tensor=tf.square(pred_Z - tf.cast(tf.reshape(LATENT_FACTORS[:,i,:,i],(-1,1)),tf.float32) ))
                        
                        # mask: n_envs x n_heads
                        # mask = tf.transpose(tf.math.equal(tf.constant(np.ones(shape=(Config.POLICY_NHEADS,Config.NUM_ENVS))*i,dtype=tf.float32) , train_model.LAB_NCE))
                        # p_seq and p_anch: n_pos_traj x 512
                        p_seq_i = get_predictor(n_out=256)(z_seq[i])

                        # z_seq_mask = tf.boolean_mask(z_seq,mask)
                        # z_anch_mask = tf.boolean_mask(z_anch,mask)
                        
                        byol_loss = ( cos_loss(p_seq_0,  z_seq[i]) + cos_loss(p_seq_i, z_seq[0]) ) / 2. + ( cos_loss(p_seq_i,  z_anch) + cos_loss(p_anch, z_seq[i]) ) / 2.
                        self.rep_loss = self.rep_loss + 1/(Config.POLICY_NHEADS-1) * byol_loss
                        
        elif Config.AGENT == 'ppo_rnd':
            # with tf.variable_scope("model", reuse=True) as scope:
            """
            RND loss
            """
            self.rnd_target = get_rnd_predictor(trainable=False)(self.h)
            self.rnd_pred = get_rnd_predictor(trainable=True)(self.h)

            self.rnd_diff = tf.square(tf.stop_gradient(self.rnd_target) - self.rnd_pred)
            self.rnd_diff = tf.reduce_mean(self.rnd_diff,1)
            rnd_diff_no_grad = tf.stop_gradient(self.rnd_diff)

        elif Config.AGENT == 'ppo_goal':
            X_T = REP_PROC[:-1,]
            X_T_1 = REP_PROC[1:,]
            X_T = tf.reshape(X_T, [-1, 64, 64, 3])
            X_T_1 = tf.reshape(X_T_1, [-1, 64, 64, 3])
            with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
                act_condit, act_invariant, _, _ = choose_cnn(X_T)
                self.h_codes =  tf.concat([act_condit, act_invariant], axis=1)
            # 512 prototypes of size 128 as per Proto-RL paper: 
            # Note that we pass in the current cluster as a one-hot encoding vector
            with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
                online_projector = get_linear_layer()
                z_t = online_projector(self.h_codes)
                
            online_predictor = get_online_predictor()
            self.u_t = online_predictor(z_t)
                
            with tf.compat.v1.variable_scope("target", reuse=tf.compat.v1.AUTO_REUSE):
                act_condit, act_invariant, _, _ = choose_cnn(X_T_1)
                target_encoder =  tf.concat([act_condit, act_invariant], axis=1)
                target_projector = get_linear_layer()
                self.z_t_1 = target_projector(target_encoder)
            self.z_t_1 = tf.linalg.normalize(self.z_t_1, ord='euclidean')[0]
            self.codes = sinkhorn(scores=tf.linalg.matmul(tf.stop_gradient(self.z_t_1), tf.linalg.normalize(self.protos, ord='euclidean')[0]))
                
        if Config.AGENT == 'ppg':
            with tf.compat.v1.variable_scope("pi_branch", reuse=tf.compat.v1.AUTO_REUSE):
                self.pd_train = [self.pdtype.pdfromlatent(self.h_pi, init_scale=0.01)[0]]
               
            with tf.compat.v1.variable_scope("model_0", reuse=tf.compat.v1.AUTO_REUSE):  
                self.vf_train = [fc(self.h, 'v_0', 1)[:, 0] ]

                # Plain Dropout version: Only fast updates / stochastic latent for VIB
                self.pd_run = self.pd_train
                self.vf_run = self.vf_train
                

                # For Dropout: Always change layer, so slow layer is never used
                self.run_dropout_assign_ops = []
        else:
            with tf.compat.v1.variable_scope("model_0", reuse=tf.compat.v1.AUTO_REUSE):
                if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
                    self.pd_train = []
                    for i in range(Config.POLICY_NHEADS):
                        with tf.compat.v1.variable_scope("head_"+str(i), reuse=tf.compat.v1.AUTO_REUSE):
                            self.pd_train.append(self.pdtype.pdfromlatent(self.h, init_scale=0.01)[0])
                    with tf.compat.v1.variable_scope("head_i", reuse=tf.compat.v1.AUTO_REUSE):
                        self.pd_train_i = self.pdtype.pdfromlatent(self.phi_STATE, init_scale=0.01)[0]
                else:
                    with tf.compat.v1.variable_scope("head_0", reuse=tf.compat.v1.AUTO_REUSE):
                        self.pd_train = [self.pdtype.pdfromlatent(self.h, init_scale=0.01)[0]]
                
                if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
                    # self.vf_train = [fc(self.h, 'v'+str(i), 1)[:, 0] for i in range(Config.POLICY_NHEADS)]
                    self.vf_train = [fc(self.h, 'v_0', 1)[:, 0] ]
                else:
                    self.vf_train = [fc(self.h, 'v_0', 1)[:, 0] ]
                if Config.AGENT == 'ppo_rnd' or Config.AGENT == 'ppo_diayn' or Config.AGENT == 'ppo_goal':
                    self.vf_i_train = fc(self.h, 'v_i', 1)[:, 0]
                    self.vf_i_run = self.vf_i_train
                if  (Config.CUSTOM_REP_LOSS and Config.AGENT == 'ppo'):
                    self.vf_i_train = fc(self.phi_STATE, 'v_i', 1)[:, 0]
                    self.vf_i_run = self.vf_i_train

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
            if Config.AGENT == 'ppo_rnd':
                a, v, v_i, r_i, neglogp = sess.run([a0_run[0], self.vf_run[0], self.vf_i_run, self.rnd_diff, neglogp0_run[0]], {X: ob})
                return a, v, v_i, r_i, self.initial_state, neglogp
            elif Config.AGENT == 'ppo_diayn':
                a, v, v_i, neglogp, r_i  = sess.run([a0_run[0], self.vf_run[0], self.vf_i_run, neglogp0_run[0], self.skill_log_prob], {X: ob, Z_INT: skill_idx, Z: one_hot_skill})
                return a, v, v_i, r_i, self.initial_state, neglogp
            elif Config.AGENT == 'ppo_goal':
                a, v, v_i, neglogp = sess.run([a0_run[0], self.vf_run[0], self.vf_i_run, neglogp0_run[0]], {X: ob, Z: one_hot_skill})
                return a, v, v_i, self.initial_state, neglogp
            elif Config.AGENT == 'ppo' and not Config.CUSTOM_REP_LOSS:
                head_idx = 0
                a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob})
                return a, v, self.initial_state, neglogp
            elif Config.AGENT == 'ppg':
                head_idx = 0
                a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob, self.X_pi: ob})
                return a, v, self.initial_state, neglogp
            else:
                # a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run, neglogp0_run[head_idx]], {X: ob})
                td_map = {**nce_dict, **{X: ob}}
                rets = sess.run(a0_run + self.vf_run + neglogp0_run + ([self.vf_i_run, self.rep_loss] if len(nce_dict) else []),td_map)
                a = rets[:len(self.pd_train)]
                v = rets[len(self.pd_train):(len(self.pd_train)+len(self.vf_train))]
                neglogp = rets[(len(self.pd_train)+len(self.vf_train)):]
                return a, v, self.initial_state, neglogp

        def rep_vec(ob, *_args, **_kwargs):
            return sess.run(self.h, {X: ob})

        def value(ob, update_frac, one_hot_skill=None, *_args, **_kwargs):
            if Config.AGENT == 'ppo_diayn' or Config.AGENT == 'ppo_goal':
                return sess.run(self.vf_run, {X: ob, Z: one_hot_skill})
            else:
                return sess.run(self.vf_run, {X: ob})

        def value_i(ob, update_frac, one_hot_skill=None, *_args, **_kwargs):
            if Config.AGENT == 'ppo_diayn' or Config.AGENT == 'ppo_goal':
                return sess.run(self.vf_i_run, {X: ob, Z: one_hot_skill})
            else:
                 return sess.run(self.vf_i_run, {self.STATE: ob, X:ob})

        def nce_fw_pass(nce_dict):
            return sess.run([self.vf_i_run,self.rep_loss],nce_dict)

        def custom_train(ob, rep_vecs):
            return sess.run([self.rep_loss], {X: ob, REP_PROC: rep_vecs})[0]
        
        def compute_codes(ob):
            return sess.run([self.codes, self.u_t, self.z_t_1], {REP_PROC: ob})

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
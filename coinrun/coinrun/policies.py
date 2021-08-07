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



class CnnPolicy(object):
	def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, max_grad_norm, override_agent=None, **conv_kwargs): #pylint: disable=W0613
		self.override_agent = override_agent
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
			# trajectories of length m, for N policy heads.
			self.STATE = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
			self.STATE_NCE = tf.compat.v1.placeholder(tf.float32, [Config.REP_LOSS_M,1,None,64,64,3])
			self.ANCH_NCE = tf.compat.v1.placeholder(tf.float32, [None,64,64,3])
			# labels of Q value quantile bins
			self.LAB_NCE = tf.compat.v1.placeholder(tf.float32, [Config.POLICY_NHEADS,None])
			self.A_i = self.pdtype.sample_placeholder([None,Config.REP_LOSS_M,1],name='A_i')
			self.R_cluster = tf.compat.v1.placeholder(tf.float32, [None, Config.NUM_ENVS])
			self.A_cluster = self.pdtype.sample_placeholder([None, Config.NUM_ENVS], name='A_cluster')
			
		if Config.AGENT == 'ppo_goal':
			# fetch ob_t from joint observations for step
			X = REP_PROC
			with tf.compat.v1.variable_scope("target", reuse=tf.compat.v1.AUTO_REUSE):
				act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(X)
				self.train_dropout_assign_ops = fast_dropout_assigned_ops
				self.run_dropout_assign_ops = slow_dropout_assign_ops
			with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
				self.h =  tf.concat([act_condit, act_invariant], axis=1)
			#     act_condit, act_invariant, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(X)
			#     self.train_dropout_assign_ops = fast_dropout_assigned_ops
			#     self.run_dropout_assign_ops = slow_dropout_assign_ops
			#     self.h =  tf.concat([act_condit, act_invariant], axis=1)
			
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
				self.adv_pi = get_linear_layer(n_in=256+15,n_out=1)(tf.concat([self.h_pi,act_one_hot],axis=1))
				self.v_pi = get_linear_layer(n_in=256,n_out=1)(self.h_pi)
		elif Config.AGENT == 'ppg_ssl':
			self.X_pi, self.processed_x_pi = observation_input(ob_space, None)
			with tf.compat.v1.variable_scope("pi_branch", reuse=tf.compat.v1.AUTO_REUSE):
				act_condit_pi, act_invariant_pi, _, _ = choose_cnn(processed_x,prefix='')
				self.h_pi =  tf.concat([act_condit_pi, act_invariant_pi], axis=1)
				act_one_hot = tf.reshape(tf.one_hot(self.A,ac_space.n), (-1,ac_space.n))
				self.adv_pi = get_linear_layer(n_in=256+15,n_out=1)(tf.concat([self.h_pi,act_one_hot],axis=1))
				self.v_pi = get_linear_layer(n_in=256,n_out=1)(self.h_pi)

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
		elif Config.AGENT == 'ppo_curl':
			"""
			CURL implementation, variable names model psuedocode described in paper
			"""
			Curl_W = tf.compat.v1.Variable(initial_value=tf.random.normal(shape=(1024, 1024)), trainable=True, name='Curl_W_Matrix')
			resized_x = tf.image.resize(processed_x, [84, 84])
			
			with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
				# resize and randomly crop observations. 
				x_q = tf.image.random_crop(resized_x, [1024, 64, 64, 3])
				self.total_batch = x_q
				z_q_1, z_q_2, _, _ = choose_cnn(x_q)
				z_q = tf.concat([z_q_1, z_q_2], axis=1)

			with tf.compat.v1.variable_scope("target", reuse=tf.compat.v1.AUTO_REUSE):
				# resize and produce different randomly crop observations. 
				x_k = tf.image.random_crop(resized_x, [1024, 64, 64, 3])
				z_k_1, z_k_2, _, _ = choose_cnn(x_k)
				z_k = tf.stop_gradient(tf.concat([z_k_1, z_k_2], axis=1))
			
			proj_k = tf.linalg.matmul(tf.transpose(z_k), Curl_W)
			logits = tf.linalg.matmul(z_q, proj_k)
			curl_logits = logits - tf.reduce_max(logits, axis=1)
			curl_labels = tf.range(tf.shape(logits)[0])
			self.curl_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=curl_labels, logits=curl_logits)
		elif Config.AGENT == 'ppg_ssl':
			"""
			MYOW part
			"""
			# y_online = self.h_pi
			# y_target = tf.stop_gradient(self.h)
			# act_one_hot = tf.reshape(tf.one_hot(self.A,ac_space.n), (-1,ac_space.n))
			
			# y_online = tf.squeeze(tf.squeeze(FiLM(widths=[256,256], name='FiLM_layer')([tf.expand_dims(tf.expand_dims(y_online,1),1), act_one_hot]),1),1)

			# dist = _compute_distance(y_online, y_online)
			# k_t = 5
			# vals, indx = tf.nn.top_k(-dist, k_t+1,sorted=True)
			
			# # N_target = y_target
			# with tf.compat.v1.variable_scope("pi_branch", reuse=tf.compat.v1.AUTO_REUSE):
			#     v_online_net = get_predictor(n_out=256)
			#     v_target_net = get_predictor(n_out=256)
			#     v_online = v_online_net(y_online)

			self.rep_loss = tf.constant([0.],dtype=tf.float32)
			# for k in range(k_t):
			#     indx2 = indx[:,k+1]
			#     N_target = tf.gather(y_target, indx2)
			#     v_target = v_target_net(N_target)
			#     r_online = get_linear_layer(n_out=256)(v_online)
			#     r_target = get_linear_layer(n_out=256)(v_target)

			#     self.rep_loss += ( tf.reduce_mean(cos_loss(r_online, v_target) + cos_loss(r_target, v_online)) ) / k_t

			"""
			Clustering part
			"""
			num_steps = 10
			SK_INPUT = REP_PROC[:num_steps]
			obs_cluster = tf.reshape(SK_INPUT, [-1, 64, 64, 3])
			
			with tf.compat.v1.variable_scope("pi_branch", reuse=tf.compat.v1.AUTO_REUSE):
				act_condit, act_invariant, _, _ = choose_cnn(obs_cluster)
				# h_codes: n_batch x n_t x n_rkhs
				self.h_codes =  tf.transpose(tf.reshape(tf.concat([act_condit, act_invariant], axis=1),[num_steps,Config.NUM_ENVS,-1]),(1,0,2))
				tblock = get_time_conv() #TemporalBlock(128, 2, 1, 1, 0.)
				seq = tf.squeeze(tblock(self.h_codes),1)
				z_t = get_linear_layer(n_in=128,n_out=128)(seq)
				
				self.u_t = get_online_predictor()(z_t)
				
			with tf.compat.v1.variable_scope("target", reuse=tf.compat.v1.AUTO_REUSE):
				act_condit, act_invariant, _, _ = choose_cnn(obs_cluster)
				target_h_codes =  tf.transpose(tf.reshape(tf.concat([act_condit, act_invariant], axis=1),[num_steps,Config.NUM_ENVS,-1]),(1,0,2))
				tblock = get_time_conv() #TemporalBlock(128, 2, 1, 1, 0.)
				target_seq = tf.squeeze(tblock(target_h_codes),1)
				self.z_t_1 = get_linear_layer(n_in=128,n_out=128)(target_seq)
			self.z_t_1 = tf.linalg.normalize(self.z_t_1, ord='euclidean')[0]
			self.codes = sinkhorn(scores=tf.linalg.matmul(tf.stop_gradient(self.z_t_1), tf.linalg.normalize(self.protos, ord='euclidean')[0]))
						
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
			"""
			Clustering part
			"""
			obs_cluster = tf.reshape(REP_PROC, [-1, 64, 64, 3])

			with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
				# h_codes: n_batch x n_t x n_rkhs
				act_condit, act_invariant, _, _ = choose_cnn(obs_cluster)
				self.h_codes =  tf.transpose(tf.reshape(tf.concat([act_condit, act_invariant], axis=1),[-1,Config.NUM_ENVS,256]),(1,0,2))
				h_t = self.h_codes[:,:-1]
				h_tp1 = self.h_codes[:,1:]
				
				# h_a_t = tf.transpose(tf.reshape(get_predictor(n_in=ac_space.n,n_out=256,prefix="SH_a_emb")( act_one_hot), (-1,Config.NUM_ENVS,256)), (1,0,2))
				h_seq = tf.reshape( tf.concat([h_t,h_tp1],2), (-1,256*2))
				if not Config.CLUSTER_CONDIT_POLICY:
					act_one_hot = tf.reshape(tf.one_hot(self.A_cluster,ac_space.n), (-1,ac_space.n))
					h_seq = tf.squeeze(tf.squeeze(FiLM(widths=[512,512], name='FiLM_layer')([tf.expand_dims(tf.expand_dims(h_seq,1),1), act_one_hot]),1),1)
				self.z_t = get_online_predictor(n_in=256*2,n_out=CLUSTER_DIMS,prefix='SH_z_pred')(h_seq)
				
				self.u_t = get_predictor(n_in=CLUSTER_DIMS,n_out=CLUSTER_DIMS,prefix='SH_u_pred')(self.z_t)
				
			self.z_t_1 = self.z_t
			# scores: n_batch x n_clusters
			scores = tf.linalg.matmul(tf.linalg.normalize(self.z_t_1, axis=1, ord='euclidean')[0], tf.linalg.normalize(self.protos, axis=1, ord='euclidean')[0])
			self.codes = sinkhorn(scores=scores)

			
			if Config.MYOW:
				"""
				Compute average cluster reward 1/N_i \sum_{C_i} V^pi(s_j)
				"""
				cluster_idx = tf.argmax(scores,1)  
				if False:
					reward_scale = []
					for i in range(Config.N_SKILLS):
						filter_ = tf.cast(tf.fill(tf.shape(self.R_cluster), i),tf.float32)
						mask = tf.cast(tf.math.equal(filter_ , self.codes),tf.float32)
						rets_cluster = tf.reduce_mean(mask * self.R_cluster)
						reward_scale.append( rets_cluster )
					self.cluster_returns = tf.stack(reward_scale)
					# Predict the average cluster value from the prototype (centroid)
					with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
						self.cluster_value_mse_loss = tf.reduce_mean( (get_predictor(n_in=CLUSTER_DIMS,n_out=1)(tf.transpose(self.protos)) - self.cluster_returns)**2 )
				else:
					self.cluster_value_mse_loss = 0.

				"""
				MYOW where k-NN neighbors are replaced by Sinkhorn clusters
				"""
				with tf.compat.v1.variable_scope("target", reuse=tf.compat.v1.AUTO_REUSE):
					# h_codes: n_batch x n_t x n_rkhs
					act_condit_target, act_invariant_target, _, _ = choose_cnn(obs_cluster)
					h_codes_target =  tf.transpose(tf.reshape(tf.concat([act_condit_target, act_invariant_target], axis=1),[-1,Config.NUM_ENVS,256]),(1,0,2))
					h_t_target = h_codes_target[:,:-1]
					h_tp1_target = h_codes_target[:,1:]
					
					# h_a_t = tf.transpose(tf.reshape(get_predictor(n_in=ac_space.n,n_out=256,prefix="SH_a_emb")( act_one_hot), (-1,Config.NUM_ENVS,256)), (1,0,2))
					h_seq_target = tf.reshape( tf.concat([h_t_target,h_tp1_target],2), (-1,256*2))
					if not Config.CLUSTER_CONDIT_POLICY:
						act_one_hot_target = tf.reshape(tf.one_hot(self.A_cluster,ac_space.n), (-1,ac_space.n))
						h_seq_target = tf.squeeze(tf.squeeze(FiLM(widths=[512,512], name='FiLM_layer')([tf.expand_dims(tf.expand_dims(h_seq_target,1),1), act_one_hot_target]),1),1)
				y_online = h_seq
				y_target = tf.stop_gradient(h_seq_target)
				

				# get K closest vectors by Sinkhorn scores
				dist = _compute_distance(y_online,y_online)
				# dist = _compute_distance(y_online,y_online)
				k_t = 3
				vals, indx = tf.nn.top_k(-dist, k_t+1,sorted=True)
				
				# N_target = y_target
				with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
					v_online_net = get_predictor(n_in=256*2,n_out=HIDDEN_DIMS_SSL,prefix='MYOW_v_pred')
					r_online_net = get_predictor(n_in=HIDDEN_DIMS_SSL,n_out=HIDDEN_DIMS_SSL,prefix='MYOW_r_pred')
					v_online = v_online_net(y_online)
					r_online = r_online_net(v_online)
				with tf.compat.v1.variable_scope("target", reuse=tf.compat.v1.AUTO_REUSE):
					v_target_net = get_predictor(n_in=256*2,n_out=HIDDEN_DIMS_SSL,prefix='MYOW_v_pred')
					r_target_net = get_predictor(n_in=HIDDEN_DIMS_SSL,n_out=HIDDEN_DIMS_SSL,prefix='MYOW_r_pred')

				self.myow_loss = 0.
				for k in range(k_t):
					indx2 = indx[:,k+1]
					N_target = tf.gather(y_target, indx2)
					v_target = v_target_net(N_target)
					r_target = r_target_net(v_target)

					self.myow_loss += ( tf.reduce_mean(cos_loss(r_online, v_target)) + tf.reduce_mean(cos_loss(r_target, v_online)) ) / k_t

				with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
					phi_s = get_online_predictor(n_in=256,n_out=CLUSTER_DIMS,prefix='SH_z_pred')(tf.reshape(h_tp1,(-1,256)))
					self.myow_loss += tf.reduce_mean(cos_loss(phi_s, tf.transpose(tf.gather(self.protos,cluster_idx,axis=1),(1,0)) ))

				self.myow_loss += self.cluster_value_mse_loss

			"""
			Intrinsic rewards
			"""
			with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
				self.R_I_SCALE = tf.nn.relu(get_linear_layer(n_in=256,n_out=1,prefix='r_i_scale',init=initializers.RandomNormal(stddev=0.11))(tf.reshape(tf.stop_gradient(h_tp1),(-1,256))))

				# self.h = get_predictor(n_in=256+Config.N_SKILLS,n_out=256)(tf.concat([self.h,tf.stop_gradient(scores)],1))

			"""
			Condition on soft-cluster assignments for policy head (Cluster Conditioned Policy )
			"""
			if Config.CLUSTER_CONDIT_POLICY:
				self.h = self.h[Config.NUM_ENVS:]
				concat_code_1 = tf.compat.v1.Variable(initial_value=tf.random.normal(shape=(32, Config.N_SKILLS)), trainable=False, name='Code_Var_step')
				concat_code_2 = tf.compat.v1.Variable(initial_value=tf.random.normal(shape=(1024, Config.N_SKILLS)), trainable=False, name='Code_Var_train')
				def live_codes():
					with tf.control_dependencies([tf.assign(concat_code_1, tf.stop_gradient(tf.reshape(self.codes, [-1, Config.N_SKILLS])))]):
						return tf.identity(concat_code_1)
				def computed_codes():
					with tf.control_dependencies([tf.assign(concat_code_2, CODES)]):
						return tf.identity(concat_code_2)
				self.concat_code = tf.cond(STEP_BOOL, live_codes, computed_codes)
				
				# print(self.h)
				# print(concat_code)
				#h_seq = tf.squeeze(tf.squeeze(FiLM(widths=[512,512], name='FiLM_layer')([tf.expand_dims(tf.expand_dims(h_seq,1),1), act_one_hot]),1),1)
			else:
				concat_code = tf.zeros([1,Config.N_SKILLS])
				
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
		elif Config.AGENT == 'ppg_ssl':
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
			with tf.compat.v1.variable_scope("online", reuse=tf.compat.v1.AUTO_REUSE):
				if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
					self.pd_train = []
					for i in range(Config.POLICY_NHEADS):
						with tf.compat.v1.variable_scope("head_"+str(i), reuse=tf.compat.v1.AUTO_REUSE):
							self.pd_train.append(self.pdtype.pdfromlatent(self.h, init_scale=0.01)[0])
					with tf.compat.v1.variable_scope("head_i", reuse=tf.compat.v1.AUTO_REUSE):
						self.pd_train_i = self.pdtype.pdfromlatent(self.h, init_scale=0.01)[0]
				elif Config.AGENT == 'ppo' or override_agent == 'ppo':
					with tf.compat.v1.variable_scope("head_0", reuse=tf.compat.v1.AUTO_REUSE):
						self.pd_train = [self.pdtype.pdfromlatent(tf.concat(self.h, axis=1), init_scale=0.01)[0]]
				else:
					with tf.compat.v1.variable_scope("head_0", reuse=tf.compat.v1.AUTO_REUSE):
						self.pd_train = [self.pdtype.pdfromlatent(tf.concat([self.h, self.concat_code], axis=1), init_scale=0.01)[0]]
				
				if Config.CUSTOM_REP_LOSS and Config.POLICY_NHEADS > 1:
					# self.vf_train = [fc(self.h, 'v'+str(i), 1)[:, 0] for i in range(Config.POLICY_NHEADS)]
					self.vf_train = [fc(self.h, 'v_0', 1)[:, 0] ]
				else:
					self.vf_train = [fc(self.h, 'v_0', 1)[:, 0] ]
				if Config.AGENT == 'ppo_rnd' or Config.AGENT == 'ppo_diayn' or Config.AGENT == 'ppo_goal':
					self.vf_i_train = fc(tf.stop_gradient(self.h), 'v_i', 1)[:, 0]
					self.vf_i_run = self.vf_i_train
				if  (Config.CUSTOM_REP_LOSS and Config.AGENT == 'ppo'):
					self.vf_i_train = fc(self.h, 'v_i', 1)[:, 0]
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
				if Config.CLUSTER_CONDIT_POLICY:
					# for step, pass in dummy values for code placeholder since we compute the codes live
					a, v, v_i, neglogp, h, h_codes, ht, htp1, ccode = sess.run([a0_run[0], self.vf_run[0], self.vf_i_run, neglogp0_run[0], self.h, self.h_codes, h_t, h_tp1, self.concat_code], {REP_PROC: ob, Z: one_hot_skill, CODES: np.zeros(shape=(1024, Config.N_SKILLS)), STEP_BOOL: True})
					return a, v, v_i, self.initial_state, neglogp,  h, h_codes, ht, htp1, ccode
				else:
					a, v, v_i, neglogp = sess.run([a0_run[0], self.vf_run[0], self.vf_i_run, neglogp0_run[0]], {REP_PROC: ob, Z: one_hot_skill})
					return a, v, v_i, self.initial_state, neglogp
			elif (Config.AGENT == 'ppo' and not Config.CUSTOM_REP_LOSS) or self.override_agent == 'ppo':
				head_idx = 0
				a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob})
				return a, v, self.initial_state, neglogp
			elif Config.AGENT == 'ppg':
				head_idx = 0
				a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob, self.X_pi: ob})
				return a, v, self.initial_state, neglogp
			elif Config.AGENT == 'ppg_ssl':
				head_idx = 0
				a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob, self.X_pi: ob})
				return a, v, self.initial_state, neglogp
			elif Config.AGENT == 'ppo_curl':
				head_idx = 0
				a, v, neglogp = sess.run([a0_run[head_idx], self.vf_run[head_idx], neglogp0_run[head_idx]], {X: ob})
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
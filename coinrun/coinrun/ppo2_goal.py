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


# k: k parameter for K-nn
def sinkhorn(scores, temp=0.1, k=3):
	def remove_infs(x):
		m = tf.math.reduce_max(x[tf.math.is_inf(x)])
		casted_x = tf.cast(tf.ones_like(x, dtype=tf.dtypes.float32), tf.float32)
		max_tensor = tf.math.multiply(casted_x, m)
		return tf.where(tf.math.is_inf(x), max_tensor, x)

	Q = scores / temp
	Q -= tf.math.reduce_max(Q)

	Q = tf.transpose(tf.math.exp(Q))
	Q /= tf.math.reduce_sum(Q)

	r = tf.ones(Q.get_shape()[0]) / tf.cast(Q.get_shape()[0], tf.float32)
	c = tf.ones(Q.get_shape()[1]) / tf.cast(Q.get_shape()[1], tf.float32)

	for it in range(k):
		u = tf.reduce_sum(Q, axis=1)
		u = tf.cast(u, tf.float32)
		u = remove_infs(r / u)
		Q = tf.cast(Q, tf.float32)
		Q *= tf.expand_dims(u, axis=1)
		Q *= tf.expand_dims((c / tf.math.reduce_sum(Q, axis=0)), axis=0)
	Q = Q / tf.math.reduce_sum(Q, axis=0, keepdims=True)
	return tf.transpose(Q)
"""
Intrinsic advantage methods
"""
def soft_update(source_variables,target_variables,tau=1.0):
	for (v_s, v_t) in zip(source_variables, target_variables):
		v_t.shape.assert_is_compatible_with(v_s.shape)
		v_t.assign((1 - tau) * v_t + tau * v_s)

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

		self.running_stats_s = RunningStats()
		self.running_stats_s_ = RunningStats()
		self.running_stats_r = RunningStats()
		self.running_stats_r_i = RunningStats()

		sess = tf.compat.v1.get_default_session()

		train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, max_grad_norm)
		act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, max_grad_norm)

		# in case we don't use rep loss
		rep_loss = None
		SKILLS = tf.compat.v1.placeholder(tf.float32, shape=[nbatch_train, Config.N_SKILLS], name='mb_skill')
		A = train_model.pdtype.sample_placeholder([None])
		U_T = tf.compat.v1.placeholder(tf.float32, [8192, 128])
		Z_T_1 = tf.compat.v1.placeholder(tf.float32, [8192, 128])
		ADV = tf.compat.v1.placeholder(tf.float32, [None])
		ADV_2 = tf.compat.v1.placeholder(tf.float32, [None])
		ADV = ADV + ADV_2
		R = tf.compat.v1.placeholder(tf.float32, [None])
		R_i = tf.compat.v1.placeholder(tf.float32, [None])
		OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [None])
		OLDVPRED = tf.compat.v1.placeholder(tf.float32, [None])
		OLDVPRED_i = tf.compat.v1.placeholder(tf.float32, [None])
		LR = tf.compat.v1.placeholder(tf.float32, [])
		CLIPRANGE = tf.compat.v1.placeholder(tf.float32, [])
		# VF loss
		vpred = train_model.vf_train  # Same as vf_run for SNI and default, but noisy for SNI2 while the boostrap is not
		vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf_train - OLDVPRED, - CLIPRANGE, CLIPRANGE)
		vf_losses1 = tf.square(vpred - R)
		vf_losses2 = tf.square(vpredclipped - R)
		vf_loss = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1, vf_losses2))

		vpred_i = train_model.vf_i_train  # Same as vf_run for SNI and default, but noisy for SNI2 while the boostrap is not
		vpredclipped_i = OLDVPRED_i + tf.clip_by_value(vpred_i - OLDVPRED_i, - CLIPRANGE, CLIPRANGE)
		vf_losses1_i = tf.square(vpred_i - R_i)
		vf_losses2_i = tf.square(vpredclipped_i - R_i)
		vf_loss_i = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1_i, vf_losses2_i))

		neglogpac_train = train_model.pd_train[0].neglogp(A)
		ratio_train = tf.exp(OLDNEGLOGPAC - neglogpac_train)
		pg_losses_train = -ADV * ratio_train
		pg_losses2_train = -ADV * tf.clip_by_value(ratio_train, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
		pg_loss = tf.reduce_mean(input_tensor=tf.maximum(pg_losses_train, pg_losses2_train))
		approxkl_train = .5 * tf.reduce_mean(input_tensor=tf.square(neglogpac_train - OLDNEGLOGPAC))
		clipfrac_train = tf.reduce_mean(input_tensor=tf.cast(tf.greater(tf.abs(ratio_train - 1.0), CLIPRANGE), dtype=tf.float32))

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

		p_t = tf.nn.log_softmax(tf.linalg.matmul(U_T, train_model.protos) / 0.1, axis=1)
		curr_codes = sinkhorn(scores=tf.linalg.matmul(Z_T_1, tf.linalg.normalize(train_model.protos, ord='euclidean')[0]))
		rep_loss = -tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.stop_gradient(curr_codes) * p_t, axis=1))
		loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * Config.L2_WEIGHT + beta * info_loss + (rep_loss*Config.REP_LOSS_WEIGHT + vf_loss_i*vf_coef)
		aux_loss = (rep_loss*Config.REP_LOSS_WEIGHT + vf_loss_i*vf_coef)
		# import ipdb;ipdb.set_trace()

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
		tot_norm = tf.zeros((1,))
		for g,v in grads_and_var:
			tot_norm += tf.norm(g)
		tot_norm = tf.reshape(tot_norm, [])
		_train = trainer.apply_gradients(grads_and_var)



		grads_and_var_aux = trainer.compute_gradients(aux_loss, params)
		grads_aux, var_aux = zip(*grads_and_var_aux)
		if max_grad_norm is not None:
			grads_aux, _grad_norm_aux = tf.clip_by_global_norm(grads_aux, max_grad_norm)
		grads_and_var_aux = list(zip(grads_aux, var_aux))
		_train_aux = trainer.apply_gradients(grads_and_var_aux)

		
		
		def train(lr, cliprange, states_nce, anchors_nce, labels_nce, u_t, z_t_1, obs, returns, returns_i, masks, actions, values, values_i, skills, neglogpacs,  states=None, train_target='policy'):
			advs = returns - values
			adv_mean = np.mean(advs, axis=0, keepdims=True)
			adv_std = np.std(advs, axis=0, keepdims=True)
			advs = (advs - adv_mean) / (adv_std + 1e-8)
			advs_i = returns_i - values_i
			adv_mean_i = np.mean(advs_i, axis=0, keepdims=True)
			adv_std_i = np.std(advs_i, axis=0, keepdims=True)
			advs_i = (advs_i - adv_mean_i) / (adv_std_i + 1e-8)

			td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr, CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values, train_model.STATE:obs, ADV_2:advs_i, OLDVPRED_i:values_i, R_i:returns_i, train_model.Z: skills, U_T: u_t, Z_T_1: z_t_1}
			if states is not None:
				td_map[train_model.S] = states
				td_map[train_model.M] = masks
			
			if train_target=='policy':
				return sess.run(
						[pg_loss, vf_loss, entropy, approxkl_train, clipfrac_train, approxkl_run, clipfrac_run, l2_loss, info_loss, rep_loss, vf_loss_i, _train],
						td_map
					)[:-1]
			elif train_target=='clustering':
				return sess.run(
						[rep_loss, _train_aux],
						td_map
					)[:-1]
			
		self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl_train', 'clipfrac_train', 'approxkl_run', 'clipfrac_run', 'l2_loss', 'info_loss_cv', 'value_i_loss', 'gradient_norm']

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
		self.value_i = act_model.value_i
		self.compute_codes = act_model.compute_codes

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
		self.eval_env = eval_env
		# create one-hot encoding array for all possible skills / codes
		a = np.array([x for x in range(Config.N_SKILLS)])
		self.one_hot_skills = np.zeros((a.size, a.max()+1))
		self.one_hot_skills[np.arange(a.size), a] = 1


	def run(self, update_frac, z, pretrain=False):
		# Here, we init the lists that will contain the mb of experiences
		mb_obs, mb_rewards, mb_actions, mb_values, mb_values_i, mb_rewards_i, mb_dones, mb_neglogpacs, mb_infos, mb_u_t, mb_z_t_1 = [],[],[],[],[],[],[],[],[], [], []
		mb_states = []
		epinfos = []
		eval_epinfos = []

		head_idx_current_batch = 0 #np.random.randint(0,Config.POLICY_NHEADS,1).item()
	   

		# extract one-hot encoding for skill / code
		one_hot_skill = self.one_hot_skills[z, :]
		one_hot_skill = np.stack(Config.NUM_ENVS*[one_hot_skill])
		# the skill remains fixed for each minibatch 
		mb_skill = np.asarray([one_hot_skill]*self.nsteps, dtype=np.int32)
		# For n in range number of steps
		for _ in range(self.nsteps):
			# Given observations, get action value and neglopacs
			# We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
			actions, values, values_i, self.states, neglogpacs = self.model.step(self.obs,  update_frac, skill_idx=z, one_hot_skill=one_hot_skill)
			mb_obs.append(self.obs.copy())
			mb_actions.append(actions)
			mb_values.append(values)
			mb_values_i.append(values_i)
			mb_neglogpacs.append(neglogpacs)
			mb_dones.append(self.dones)
			

			# Take actions in env and look the results
			# Infos contains a ton of useful informations
			self.obs[:], rewards, self.dones, self.infos = self.env.step(actions)
			# eval for zero shot generalization
			eval_actions, eval_values, _, eval_states, eval_neglogpacs = self.model.step(self.eval_obs, update_frac, skill_idx=z, one_hot_skill=one_hot_skill)
			self.eval_obs[:], eval_rewards, self.eval_dones, self.eval_infos = self.eval_env.step(eval_actions)
			for info in self.infos:
				maybeepinfo = info.get('episode')
				if maybeepinfo: epinfos.append(maybeepinfo)
			
			for info in self.eval_infos:
				eval_maybeepinfo = info.get('episode')
				if eval_maybeepinfo: eval_epinfos.append(eval_maybeepinfo)
			mb_infos.append([[float(v) for k,v in info_.items() if k != 'episode'] for info_ in self.infos])
			mb_rewards.append(rewards)
 
		states_nce = rewards_nce = dones_nce = infos_nce = labels_nce = anchors_nce = tf.compat.v1.placeholder(tf.float32, [None])

		#batch of steps to batch of rollouts
		mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
		#vidwrite('plunder_avg_phi_attempt-{}.avi'.format(datetime.datetime.now().timestamp()), mb_obs[:, 0, :, :, :])
		mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
		mb_actions = np.asarray(mb_actions)
		mb_values = np.asarray(mb_values, dtype=np.float32)
		mb_values_i = np.asarray(mb_values_i, dtype=np.float32)
		mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
		mb_infos = np.asarray(mb_infos, dtype=np.float32)
		mb_dones = np.asarray(mb_dones, dtype=np.bool)	
		last_values = self.model.value(self.obs, update_frac, one_hot_skill=one_hot_skill)[0]
		last_values_i = self.model.value_i(self.obs, update_frac, one_hot_skill=one_hot_skill)
		mb_codes = []

		# compute codes
		codes, u_t, z_t_1 = self.model.compute_codes(mb_obs)

		# add in gaussian noise for missed sample.
		rand_arr_1 = np.random.normal(size=(32, 128))
		rand_arr_2 = np.random.normal(size=(32, Config.N_SKILLS))
		u_t = np.concatenate([u_t, rand_arr_1])
		z_t_1 = np.concatenate([rand_arr_1, z_t_1])
		codes = np.concatenate([rand_arr_2, codes])
		mb_codes.append(codes)
		mb_u_t.append(u_t)
		mb_z_t_1.append(z_t_1)
			
		codes = np.asarray(mb_codes)
		mb_u_t = np.asarray(mb_u_t, dtype=np.float32)
		mb_z_t_1 = np.asarray(mb_z_t_1, dtype=np.float32)
		codes = codes.reshape((-1, Config.N_SKILLS))
		mb_u_t = mb_u_t.reshape((-1, 128))
		mb_z_t_1 = mb_z_t_1.reshape((-1, 128))
		# for each observation, find the most likely code/cluster
		hard_codes = np.argmax(codes, axis=1)
		
		# mask to find states relevant to the current code
		masked_codes = np.ma.masked_equal(hard_codes, z).mask
		# rewards are 1 for states with the current code, zero else
		code_rewards = masked_codes*1
		mb_rewards_i = np.reshape(code_rewards, (256, 32))
		

		# if pretrain is true, ignore extrinsic reward
		if pretrain:
			mb_rewards = np.zeros_like(mb_rewards)
			mb_values = np.zeros_line(mb_values)
			last_values = np.zeros_like(last_values)
		# normalize reward
		# for t in range(self.nsteps):    
		#     mb_rewards_i[t] = running_stats_fun(self.model.running_stats_r_i, mb_rewards_i[t], 1, False)       
		#     # mb_rewards[t] = running_stats_fun(self.model.running_stats_r, mb_rewards[t], 1, False)    

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
		return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_values_i, mb_skill, mb_neglogpacs, mb_infos, mb_u_t, mb_z_t_1)),
			states_nce, anchors_nce, labels_nce, epinfos, eval_epinfos, mb_rewards_i, last_values_i)

	def compute_intrinsic_returns(self, mb_rewards_i, mb_values_i, last_values_i, mb_dones):
		mb_returns_i = np.zeros_like(mb_rewards_i)
		mb_advs_i = np.zeros_like(mb_rewards_i)
		lastgaelam_i = 0
		for t in reversed(range(self.nsteps)):
			"""
			Intrinsic advantages
			"""
			if t == self.nsteps - 1:
				nextnonterminal_i = 1.0 - self.dones
				nextvalues_i = last_values_i
			else:
				nextnonterminal_i = 1.0 - mb_dones[t+1]
				nextvalues_i = mb_values_i[t+1]

			delta_i = mb_rewards_i[t] + self.gamma * nextvalues_i * nextnonterminal_i - mb_values_i[t]
			mb_advs_i[t] = lastgaelam_i = delta_i + self.gamma * self.lam * nextnonterminal_i * lastgaelam_i
		mb_returns_i = mb_advs_i + mb_values_i
		return sf01(mb_returns_i)


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

	runner = Runner(env=env, model=model, eval_env=eval_env, nsteps=nsteps, gamma=gamma, lam=lam)

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

	z_iter = 0
	curr_z = np.random.randint(0, high=Config.N_SKILLS)
	tb_writer = TB_Writer(sess)
	import os
	os.environ["WANDB_API_KEY"] = "02e3820b69de1b1fcc645edcfc3dd5c5079839a1"
	group_name = "%s__%s__%f" %(Config.ENVIRONMENT,Config.RUN_ID,Config.REP_LOSS_WEIGHT)
	name = "%s__%s__%f__%d" %(Config.ENVIRONMENT,Config.RUN_ID,Config.REP_LOSS_WEIGHT,np.random.randint(100000000))
	wandb.init(project='procgen_generalization', entity='ssl_rl', config=Config.args_dict, group=group_name, name=name, mode="disabled" if Config.DISABLE_WANDB else "online")
	for update in range(start_update+1, nupdates+1):
		# update momentum encoder
		params = tf.compat.v1.trainable_variables()
		source_params = [p for p in params if "online" in p.name]
		target_params = [p for p in params if "target" in p.name]
		soft_update(source_params, target_params, tau=0.95)


		assert nbatch % nminibatches == 0
		nbatch_train = nbatch // nminibatches
		tstart = time.time()
		frac = 1.0 - (update - 1.0) / nupdates
		lrnow = lr(frac)
		cliprangenow = cliprange(frac)

		mpi_print('collecting rollouts...')
		run_tstart = time.time()
		if z_iter < Config.SKILL_EPOCHS:
			packed = runner.run(update_frac=update/nupdates, z=curr_z)
			z_iter += 1
		else:
			# sample new skill/code for current episodes
			curr_z = np.random.randint(0, high=Config.N_SKILLS)
			packed = runner.run(update_frac=update/nupdates, z=curr_z)
			z_iter = 0

		obs, returns, masks, actions, values, values_i, skill, neglogpacs, infos, u_t, z_t_1,  states_nce, anchors_nce, labels_nce, epinfos, eval_epinfos, rewards_i, last_values_i, = packed
		u_t = u_t.reshape((-1, 128))
		z_t_1 = z_t_1.reshape((-1, 128))
		skill = np.reshape(skill, (-1, Config.N_SKILLS))
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
		E_ppo = noptepochs
		E_clustering = Config.GOAL_EPOCHS
		mbinds = inds[0:nbatch_train]
		dummy_slice = (arr[mbinds] for arr in (obs, returns, returns, masks, actions, values, values_i, skill, neglogpacs))

		for _ in range(E_clustering):
			rep_loss = model.train(lrnow, cliprangenow, states_nce, anchors_nce, labels_nce, u_t, z_t_1, *dummy_slice, train_target='clustering')[0]
		scale_i = np.log(1/rep_loss + 1)
		returns_i = runner.compute_intrinsic_returns(rewards_i*scale_i, values_i.reshape(-1, 32), last_values_i, masks.reshape(-1, 32))

		for _ in range(noptepochs):
			np.random.shuffle(inds)
			for start in range(0, nbatch, nbatch_train):
				sess.run([model.train_model.train_dropout_assign_ops])
				end = start + nbatch_train
				mbinds = inds[start:end]
				slices = (arr[mbinds] for arr in (obs, returns, returns_i, masks, actions, values, values_i, skill, neglogpacs))
				mblossvals.append(model.train(lrnow, cliprangenow, states_nce, anchors_nce, labels_nce,  u_t, z_t_1, *slices, train_target='policy'))
		# for _ in range(E_clustering):
		# 	np.random.shuffle(inds)
		# 	for start in range(0, nbatch, nbatch_train):
		# 		sess.run([model.train_model.train_dropout_assign_ops])
		# 		end = start + nbatch_train
		# 		mbinds = inds[start:end]
		# 		slices = (arr[mbinds] for arr in (obs, returns, returns_i, masks, actions, values, values_i, skill, neglogpacs, u_t, z_t_1))
		# 		model.train(lrnow, cliprangenow, states_nce, anchors_nce, labels_nce, *slices, train_target='clustering')
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

			mpi_print('eplenmean', ep_len_mean)
			mpi_print('eprew', rew_mean_10)
			mpi_print('eprew_eval', eval_rew_mean)
			mpi_print('fps', fps)
			mpi_print('total_timesteps', update*nbatch)
			mpi_print([epinfo['r'] for epinfo in epinfobuf10])
			
			if len(mblossvals):
				for (lossval, lossname) in zip(lossvals, model.loss_names):
					mpi_print(lossname, lossval)
					tb_writer.log_scalar(lossval, lossname, step=step)
			mpi_print('----\n')

			wandb.log({"%s/ep_len_mean"%(Config.ENVIRONMENT): ep_len_mean,
                        "%s/avg_value"%(Config.ENVIRONMENT):avg_value,
                        "%s/custom_loss"%(Config.ENVIRONMENT):mean_cust_loss,
                        "%s/eplenmean"%(Config.ENVIRONMENT):ep_len_mean,
                        "%s/eprew"%(Config.ENVIRONMENT):rew_mean_10,
                        "%s/eprew_eval"%(Config.ENVIRONMENT):eval_rew_mean,
                        "%s/rep_loss"%(Config.ENVIRONMENT):rep_loss,
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
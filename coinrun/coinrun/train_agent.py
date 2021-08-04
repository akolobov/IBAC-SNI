"""
Train an agent using a PPO2 based on OpenAI Baselines.

To run:

conda activate aidl
cd jobs/IBAC-SNI/coinrun

PSE:
python3 -m coinrun.train_agent --env coinrun --run-id pse --num-levels 0 --short --agent ppo_pse

Our algo:
python3 -m coinrun.train_agent --env coinrun --run-id baseline --num-levels 0 --short --rep_loss -n-heads 5 -m 10 -rep_lambda 1

DIAYN:
python3 -m coinrun.train_agent --env coinrun --run-id diayn --num-levels 0 --short --agent ppo_diayn -skilleps 4 -n_skills 5

CURL:
python3 -m coinrun.train_agent --env coinrun --run-id curl --num-levels 0 --short --agent ppo_curl

Goal conditioned PPO :
python3 -m coinrun.train_agent --env coinrun --run-id goal --num-levels 0 --short --agent ppo_goal -gpu 1 -n_skills 50
(with hard codes)
python3 -m coinrun.train_agent --env coinrun --run-id goal --num-levels 0 --short --agent ppo_goal -gpu 1 -n_skills 50 --hgoal
(with ema)
python3 -m coinrun.train_agent --env coinrun --run-id goal --num-levels 0 --short --agent ppo_goal -gpu 1 -n_skills 50 --ema
(with intrinsic reward)
python3 -m coinrun.train_agent --env coinrun --run-id goal --num-levels 0 --short --agent ppo_goal -gpu 1 -n_skills 50 --itr
(with mine your own view)
python3 -m coinrun.train_agent --env coinrun --run-id goal --num-levels 0 --short --agent ppo_goal -gpu 1 -n_skills 50 --myow
(with intrinsic reward decay)
python3 -m coinrun.train_agent --env coinrun --run-id goal --num-levels 0 --short --agent ppo_goal -gpu 1 -n_skills 50 -itr_decay 0.9
(default Cluster Conditioned Policy)
python3 -m coinrun.train_agent --env coinrun --run-id goal --num-levels 0 --short --agent ppo_goal -gpu 1 -n_skills 50 --ema --myow --ccp
RND:
python3 -m coinrun.train_agent --env coinrun --run-id rnd --num-levels 0 --short --agent ppo_rnd

# To run and save models on AIDL VM to path with 1 TB of storage
python3 -m coinrun.train_agent --env coinrun --run-id test --num-levels 0 --vshort --agent ppo -phase1 easy -phase2 easy -respath /mnt/saved_models/

# to change the distribution mode of the first and second phase:
-phase1 exploration -phase2 hard
-phase1 hard -phase2 exploration
# options are either 'easy', 'hard', or 'exploration'
--------------------------------------------------------------------------------------------------------------------------------
To plot (local):
tensorboard --logdir=results-procgen/tb_log/ --host localhost --port 8888

To test Procgen latent factors export:
python -c "import gym;env=gym.make('procgen:procgen-heist-v0');env.reset();print(env.step(0))"
--------------------------------------------------------------------------------------------------------------------------------

To launch on Dilbert:

conda activate dilbert
cd ~/jobs/dilbert/rl_nexus/
python jobs_launcher.py --env  bigfish bossfight caveflyer chaser climber coinrun dodgeball fruitbot heist jumper leaper maze miner ninja plunder starpilot --nheads 5 --m 5 --rep_lambda 1 --phase1 easy --phase2 None
bash extract_and_plot.sh

exploration only envs: coinrun caveflyer leaper jumper maze heist climber ninja
all envs: bigfish bossfight caveflyer chaser climber coinrun dodgeball fruitbot heist jumper leaper maze miner ninja plunder starpilot

To SSH into Dilbert node:
cd /mnt/batch/tasks/workitems/job3244/job-1/task2/wd/.xt/cwd/rl_nexus



GS envs: bigfish leaper starpilot plunder

python jobs_launcher.py --agent ppo_goal_bogdan --env bigfish leaper starpilot plunder  --nheads 200  --run_ID bogdan_smarterMYOW --n_knn 1 3 --cluster_T 10
"""
print('Importing packages')
import os
import copy
import time
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
from baselines.bench.monitor import ResultsWriter
print('Imported baselines')
from collections import deque
import coinrun.main_utils as utils
from coinrun import setup_utils, wrappers
from coinrun.config import Config
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple
from gym3.types import (
    DictType,
    Discrete,
    Real,
    TensorType,
    ValType,
    multimap,
    INTEGER_DTYPE_NAMES,
    FLOAT_DTYPE_NAMES,
)
print('Imported coinrun')
from baselines.common.mpi_util import setup_mpi_gpus
from gym.spaces import Box, Dict, Discrete as DiscreteG
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv
print('Imported procgen')

from baselines.common.vec_env import (
    VecExtractDictObs,
    VecFrameStack,
    VecNormalize,
    VecEnvWrapper
)

mpi_print = utils.mpi_print

import sys

# defined vec_monitor here for modiciations 12/17/2020
class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart},
                extra_keys=info_keywords)
        else:
            self.results_writer = None
        self.info_keywords = info_keywords
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                for k in self.info_keywords:
                    epinfo[k] = info[k]
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(epinfo)
                newinfos[i] = info
        return obs, rews, dones, newinfos

# MOD
def _vt2space(vt: ValType):
    from gym import spaces

    def tt2space(tt: TensorType):
        if isinstance(tt.eltype, Discrete):
            if tt.ndim == 0:
                return spaces.Discrete(tt.eltype.n)
            else:
                return spaces.Box(
                    low=0,
                    high=tt.eltype.n - 1,
                    shape=tt.shape,
                    dtype=types_np.dtype(tt),
                )
        elif isinstance(tt.eltype, Real):
            return spaces.Box(
                shape=tt.shape,
                dtype=types_np.dtype(tt),
                low=float("-inf"),
                high=float("inf"),
            )
        else:
            raise NotImplementedError

class FakeEnv:
    """
    Create a baselines VecEnv environment from a gym3 environment.
    :param env: gym3 environment to adapt
    """

    def __init__(self, env, obs_space, act_space):
        self.env = env
        # for some reason these two are returned as none
        # so I passed in a baselinevec object and copied
        # over it's values
        # self.baseline_vec = baselinevec
        self.observation_space = obs_space
        self.action_space = act_space
        self.rewards = np.zeros(Config.NUM_ENVS)
        self.lengths = np.zeros(Config.NUM_ENVS)
        self.aux_rewards = None
        self.long_aux_rewards = None

    def reset(self):
        _rew, ob, first = self.env.observe()
        if not first.all():
            print("Warning: manual reset ignored")
        return ob

    def step_async(self, ac):
        self.env.act(ac)

    def step_wait(self):
        rew, ob, first = self.env.observe()
        return ob, rew, first, self.env.get_info()

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    @property
    def num_envs(self):
        return self.env.num

    def render(self, mode="human"):
        # gym3 does not have a generic render method but the convention
        # is for the info dict to contain an "rgb" entry which could contain
        # human or agent observations
        info = self.env.get_info()[0]
        if mode == "rgb_array" and "rgb" in info:
            return info["rgb"]

    def close(self):
        pass
    
    # added this in to see if it'll properly call the method for the gym3 object
    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        return self.env.callmethod(method, *args, **kwargs)
# end mod


# helper function to make env
def make_env(steps_per_env):
    observation_space = Dict(rgb=Box(shape=(64,64,3),low=0,high=255))
    action_space = DiscreteG(15)
    if Config.FIRST_PHASE == 'exploration':
        # baseline_vec_train = ProcgenEnv(num_envs=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.FIRST_PHASE)
        gym3_env_train = ProcgenGym3Env(num=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.FIRST_PHASE, start_level=Config.START_LEVEL)
    else:
        # baseline_vec_train = ProcgenEnv(num_envs=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, num_levels=Config.NUM_LEVELS, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.FIRST_PHASE)
        gym3_env_train = ProcgenGym3Env(num=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, num_levels=Config.NUM_LEVELS, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.FIRST_PHASE, start_level=Config.START_LEVEL)
    if Config.SECOND_PHASE == 'exploration':
        # baseline_vec_adapt = ProcgenEnv(num_envs=Config.NUM_ENVS, env_name=Config.ENVIRONMENT,  paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.SECOND_PHASE)
        gym3_env_adapt = ProcgenGym3Env(num=Config.NUM_ENVS, env_name=Config.ENVIRONMENT,  paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.SECOND_PHASE, start_level=Config.START_LEVEL)
    elif Config.SECOND_PHASE != "None":
        # baseline_vec_adapt = ProcgenEnv(num_envs=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, num_levels=Config.NUM_LEVELS, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.SECOND_PHASE)
        gym3_env_adapt = ProcgenGym3Env(num=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, num_levels=Config.NUM_LEVELS, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.SECOND_PHASE, start_level=Config.START_LEVEL)
    else:
        baseline_vec_adapt = gym3_env_adapt = None
    
    venv_train = FakeEnv(gym3_env_train, observation_space, action_space)
    venv_train = VecExtractDictObs(venv_train, "rgb")
    if Config.SECOND_PHASE != "None":
        venv_adapt = FakeEnv(gym3_env_adapt, observation_space, action_space)   
        venv_adapt = VecExtractDictObs(venv_adapt, "rgb")
    venv_train = VecMonitor(
        venv=venv_train, filename=None, keep_buf=100,
    )
    if Config.SECOND_PHASE != "None":
        venv_adapt = VecMonitor(
            venv=venv_adapt, filename=None, keep_buf=100,
        )

    venv_train = VecNormalize(venv=venv_train, ob=False)
    venv_train = wrappers.add_final_wrappers(venv_train)
    if Config.SECOND_PHASE != "None":
        venv_adapt = VecNormalize(venv=venv_adapt, ob=False)
        venv_adapt = wrappers.add_final_wrappers(venv_adapt)

        venv = wrappers.DistributionShiftWrapperVec(env_list=[venv_train, venv_adapt], steps_per_env=steps_per_env) 
    else:
        venv = venv_train
        venv_adapt = venv_train = None
        venv.current_env_steps_left = steps_per_env

    return venv, venv_train, venv_adapt

def main():
    print('Parsing args')
    args = setup_utils.setup_and_load()
    print('Setting up MPI')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)
    print('Setting config')
    # coinrun version, allows you to specify how many GPUs you want this run to use
    #utils.setup_mpi_gpus()

    # baselines version, just sets the number of GPUs to the -n flag 
    #setup_mpi_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(Config.NUM_GPUS)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    
    total_timesteps = int(160e6)
    if Config.LONG_TRAINING:
        total_timesteps = int(200e6)
    elif Config.SHORT_TRAINING:
        total_timesteps = int(8e6)
    elif Config.VERY_SHORT_TRAINING:
        total_timesteps = int(500e3)
    save_interval = args.save_interval

    #env = utils.make_general_env(nenvs, seed=rank)
    #print (env)

    mpi_print(Config.ENVIRONMENT)
    venv, venv_train, venv_adapt = make_env(total_timesteps//2) #switch "easy" -> "exploration" halfway
    # import ipdb;ipdb.set_trace()
    observation_space = Dict(rgb=Box(shape=(64,64,3),low=0,high=255))
    action_space = DiscreteG(15)
    
    # baseline_vec_eval = ProcgenEnv(num_envs=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, num_levels=0, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.FIRST_PHASE)
    gym3_env_eval = ProcgenGym3Env(num=Config.NUM_ENVS, env_name=Config.ENVIRONMENT, num_levels=0, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode=Config.FIRST_PHASE)

    venv_eval = FakeEnv(gym3_env_eval, observation_space, action_space)
    venv_eval = VecExtractDictObs(venv_eval, "rgb")
    venv_eval = VecMonitor(
        venv=venv_eval, filename=None, keep_buf=100,
    )
    venv_eval = VecNormalize(venv=venv_eval, ob=False)
    venv_eval = wrappers.add_final_wrappers(venv_eval)

    
    with tf.compat.v1.Session(config=config) as sess:
        
        

        #sess.run(tf.compat.v1.global_variables_initializer())
        
        if Config.AGENT == 'ppo':
            from coinrun import ppo2 as agent
            from coinrun import policies
        elif Config.AGENT == 'ppo_rnd':
            from coinrun import ppo2_rnd as agent
            from coinrun import policies
        elif Config.AGENT == 'ppo_diayn':
            from coinrun import ppo2_diayn as agent
            from coinrun import policies
        elif Config.AGENT == 'ppg':
            from coinrun import ppo2_ppg as agent
            from coinrun import policies
        elif Config.AGENT == 'ppg_ssl':
            from coinrun import ppo2_ppg_ssl as agent
            from coinrun import policies
        elif Config.AGENT == 'ppo_goal':
            from coinrun import ppo2_goal as agent
            from coinrun import policies
        elif Config.AGENT == 'ppo_curl':
            from coinrun import ppo2_curl as agent
            from coinrun import policies
        elif Config.AGENT == 'ppo_goal_bogdan':
            from coinrun import ppo2_goal_bogdan as agent
            from coinrun import policies_bogdan as policies
        elif Config.AGENT == 'ppg_cluster':
            from coinrun import ppo2_ppg_sinkhorn as agent
            from coinrun import policies_ppg_sinkhorn as policies
        elif Config.AGENT == 'ppo_bisimulation':
            from coinrun import ppo2_bisimulation as agent
            from coinrun import policies_bisimulation as policies
        elif Config.AGENT == 'ppo_pse':
            from coinrun import ppo2_pse as agent
            from coinrun import policies_pse as policies
        policy = policies.get_policy()

        agent.learn(policy=policy,
                    env=venv,
                    eval_env=venv_eval,
                    save_interval=save_interval,
                    nsteps=Config.NUM_STEPS,
                    nminibatches=Config.NUM_MINIBATCHES,
                    lam=0.95,
                    gamma=Config.GAMMA,
                    noptepochs=Config.PPO_EPOCHS,
                    log_interval=1,
                    ent_coef=Config.ENTROPY_COEFF,
                    lr=lambda f : f * Config.LEARNING_RATE,
                    cliprange=lambda f : f * 0.2,
                    total_timesteps=total_timesteps)

if __name__ == '__main__':
    main()


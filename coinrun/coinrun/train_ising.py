"""

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

from coinrun import ising_env

import gym


class VecMonitor(gym.Wrapper):
    def __init__(self, venv):
        super().__init__(self, venv)
        self.eprets = 0
        self.eplens = 0
        self.epcount = 0

    def reset(self):
        obs = self.venv.reset()
        self.eprets = 0
        self.eplens = 0
        return obs.expand_dims(0)

    def step(self,a):
        obs, rews, dones, infos = self.venv.step(a)
        self.eprets += rews
        self.eplens += 1

        info = infos.copy()
        
        if dones:
            
            ret = self.eprets
            eplen = self.eplens
            epinfo = {'r': ret, 'l': eplen}
            
            info['episode'] = epinfo
            
            self.epcount += 1
            self.eprets = 0
            self.eplens = 0
            
        return obs.expand_dims(0), np.array([rews]), np.array([dones]), np.array([info])

# helper function to make env
def make_env(steps_per_env):
    venv = VecMonitor(ising_env.IsingEnv(T=32,k=5))
    return venv

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
    
    total_timesteps = int(1e6)
    
    save_interval = args.save_interval

    #env = utils.make_general_env(nenvs, seed=rank)
    #print (env)

    mpi_print(Config.ENVIRONMENT)
    venv = make_env(total_timesteps)
    venv_eval = make_env(total_timesteps)
    # import ipdb;ipdb.set_trace()
    observation_space = Dict(rgb=Box(shape=(64,64,3),low=0,high=255))
    action_space = DiscreteG(5)
    

    
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


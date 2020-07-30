"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""
import copy
import time
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ppo2
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
#from baselines.ppo2 import ppo2
#from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from gym.spaces import Box, Dict
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv

from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)

import sys

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

    def __init__(self, env, baselinevec):
        self.env = env
        # for some reason these two are returned as none
        # so I passed in a baselinevec object and copied
        # over it's values
        self.observation_space = copy.deepcopy(baselinevec.observation_space)
        self.action_space = copy.deepcopy(baselinevec.action_space)

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

def main():
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()
    #setup_mpi_gpus()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    nenvs = Config.NUM_ENVS
    
    total_timesteps = int(160e6)
    if Config.LONG_TRAINING:
        total_timesteps = int(200e6)
    elif Config.SHORT_TRAINING:
        #total_timesteps = int(120e6)
        total_timesteps = int(10e6)
    save_interval = args.save_interval

    #env = utils.make_general_env(nenvs, seed=rank)
    #print (env)

    print (Config.ENVIRONMENT)
    
    baseline_vec = ProcgenEnv(num_envs=nenvs, env_name=Config.ENVIRONMENT, num_levels=Config.NUM_LEVELS, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode="easy")
    gym3_env = ProcgenGym3Env(num=nenvs, env_name=Config.ENVIRONMENT, num_levels=Config.NUM_LEVELS, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode="easy")
    venv = FakeEnv(gym3_env, baseline_vec)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)
    
    rep_loss = False
    if Config.CUSTOM_REP_LOSS:
        rep_loss = True
    #sys.exit(0)
    with tf.Session(config=config) as sess:
        #env = wrappers.add_final_wrappers(env)
        venv = wrappers.add_final_wrappers(venv)
        
        policy = policies.get_policy()

        #sess.run(tf.global_variables_initializer())
        ppo2.learn(policy=policy,
                    env=venv,
                    #env=env,
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
                    total_timesteps=total_timesteps,
                    rep_loss_bool=rep_loss)

if __name__ == '__main__':
    main()


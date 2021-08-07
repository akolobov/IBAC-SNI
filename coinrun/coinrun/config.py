from mpi4py import MPI
import argparse
import os
import datetime

def count_latent_factors(game):
    if game == 'heist':
        return 1
    if game == 'leaper':
        return 3
    return 0

class ConfigSingle(object):
    """
    A global config object that can be initialized from command line arguments or
    keyword arguments.
    """
    def __init__(self):
        # self.WORKDIR = './results-procgen/saved_models-{}-{}-{}'.format('plunder',"test", datetime.datetime.now().timestamp())
        # self.TB_DIR =  './results-procgen/tb_log-{}-{}-{}'.format('plunder', "test", datetime.datetime.now().timestamp())
        self.WORKDIR = './results-procgen/saved_models'
        self.TB_DIR =  './results-procgen/tb_log'
        if not os.path.exists(self.WORKDIR):
            os.makedirs(self.WORKDIR, exist_ok=True)

        self.LOG_ALL_MPI = True
        self.SYNC_FROM_ROOT = True

        arg_keys = []
        bool_keys = []
        type_keys = []

        # If there are 5 values and the last one is True, it gets saved & loaded (althought I'm not sure that's working properly)

        ### Only for test_vars and enjoy:

        # Save path for model weights
        # Use '/mnt/saved_models/' for lots of storage
        type_keys.append(('respath', 'restore_path', str, './saved_models'))

        # Helper for the case that we need to replay
        bool_keys.append(('replay', 'replay'))

        # Probability to save each image when running coinrun.enjoy.
        type_keys.append(('save-images', 'save_images', float, 0))

        # The runid whose parameters and settings you want to load.
        type_keys.append(('residd', 'restore_idd', str, None))

        # For IBAC, use SNI with lambda=0.5 
        bool_keys.append(('sni', 'sni', True))

        # For dropout, use SNI with lambda=0.5 
        bool_keys.append(('sni2', 'sni2', True))

        # For exponential moving average updates on target encoder
        bool_keys.append(('ema', 'ema'))

        # For computing intrinsic reward
        bool_keys.append(('itr', 'intrinsic'))

        # For enabling a cluster conditioned Policy
        bool_keys.append(('ccp', 'cluster_condit_policy'))

        # Decay for weight on intrinsic reward. Default values anneals to 1e-4 at the end of 25M steps
        type_keys.append(('itr_decay', 'intrinsic_reward_decay', float, 0.997))

        # for Mine Your Own View SSL prediction
        bool_keys.append(('myow','myow'))
        # Sinkhorn consecutive timesteps
        type_keys.append(('cluster_t', 'cluster_t', int, 2))

        # Tempterature parameter for sinkhorn
        type_keys.append(('temp', 'temp', float, 0.1))
        ### Others:
        # Use hard cluster/code assignments for intrinsic reward
        bool_keys.append(('hgoal', 'hard_codes'))
        # Long training for 200M time steps
        bool_keys.append(('long', 'long_training'))

        # Short training for 25M time steps
        bool_keys.append(('short', 'short_training'))

        # Very short training for 5M time steps
        bool_keys.append(('vshort', 'very_short_training'))

        # Bool for using the custom representation loss, if the flag is passed then the loss will be used
        bool_keys.append(('rep_loss', 'custom_rep_loss'))

        # Bool for using joint RL + sinkhorn update
        bool_keys.append(('skrl', 'joint_skrl'))

        # Weighting value for the custom representation loss
        type_keys.append(('rep_lambda', 'rep_loss_weight', float, 0))

        # total number of skills that can be sampled for DIAYN/ Proto-RL
        type_keys.append(('n_skills', 'n_skills', int, 5))

        # Number of epochs before we switch skills for diayn or protoRL
        type_keys.append(('skilleps', 'skill_epochs', int, 2))

        type_keys.append(('n_knn', 'n_knn', int, 1))

        # Number of gradient updates made for clustering / SSL layer
        type_keys.append(('goaleps', 'goal_epochs', int, 1))

        # Distribution mode for first phase of training
        type_keys.append(('phase1', 'first_phase', str, 'hard'))

        # Distribution mode for second phase of training
        type_keys.append(('phase2', 'second_phase', str, 'exploration'))

        # The number of the GPU to be used (default is zero which should be the base GPU)
        type_keys.append(('gpu', 'num_gpus', int, 0))

        # Beta value for Info-loss KL divergence. -1 leaves this loss term out. 0 will probably diverge
        type_keys.append(('b', 'beta', float, -1., True))

        # Number of samples for MC averages when computing the loss 
        type_keys.append(('nr-samples', 'nr_samples', int, 1, True))

        # Beta value for Info-loss L2 on Activations. -1 leaves this loss term out. 0 will probably diverge
        type_keys.append(('bl2a', 'beta_l2a', float, -1., True))

        # How many train processes per test process. Defaults to 2
        type_keys.append(('test-ratio', 'test_ratio', int, 4, False))

        # Number of timesteps we want to use for custom loss
        type_keys.append(('m', 'rep_loss_m', int, 1))

        # Number of negative examples we want to use for custom loss
        type_keys.append(('negs', 'NEGS', int, 20))

        # Number of hidden units/latent space dim. Assuming representation layer is 256 units
        type_keys.append(('nodes', 'nodes', int, 256))

        # Deactivated because not needed. To re-activate, uncomment line in main_utils.py as well
        # GPU offset: Use RCALL_NUM_GPU, starting from this value
        # type_keys.append(('gpu-offset', 'gpu_offset', int, 0, False))

        # The runid, used to determine the name for save files.
        type_keys.append(('runid', 'run_id', str, 'tmp'))

        # The runid whose parameters and settings you want to load.
        type_keys.append(('resid', 'restore_id', str, None))

        # Restore number of updates
        type_keys.append(('resstep', 'restore_step', int, None))

        # The game to be played.
        # One of {'standard', 'platform', 'maze'} (for CoinRun, CoinRun-Platforms, Random-Mazes)
        type_keys.append(('gamet', 'game_type', str, 'standard', True)) 

        # The convolutional architecture to use
        # One of {'nature', 'impala', 'impalalarge'}
        type_keys.append(('arch', 'architecture', str, 'impala', True))

        type_keys.append(('env', 'environment', str, 'coinrun', True))

        type_keys.append(('n-heads', 'POLICY_NHEADS', int, 1, True))
        
        # Should the model include an LSTM
        type_keys.append(('lstm', 'use_lstm', int, 0, True))

        # The number of parallel environments to run
        type_keys.append(('ne', 'num_envs', int, 32, True))

        # The number of levels in the training set.
        # If NUM_LEVELS = 0, the training set is unbounded. All level seeds will be randomly generated.
        # Use SET_SEED = -1 and NUM_LEVELS = 500 to train with the same levels in the paper.
        type_keys.append(('nlev', 'num_levels', int, 0, True))

        type_keys.append(('start_level', 'start_level', int, 0, True))

        # Provided as a seed for training set generation.
        # If SET_SEED = -1, this seed is not used and level seeds with be drawn from the range [0, NUM_LEVELS).
        # Use SET_SEED = -1 and NUM_LEVELS = 500 to train with the same levels in the paper.
        # NOTE: This value must and will be saved, in order to use the same training set for evaluation and/or visualization.
        type_keys.append(('set-seed', 'set_seed', int, -1, True))

        # PPO Hyperparameters
        type_keys.append(('ns', 'num_steps', int, 256))
        type_keys.append(('nmb', 'num_minibatches', int, 8))
        type_keys.append(('ppoeps', 'ppo_epochs', int, 3))
        type_keys.append(('ent', 'entropy_coeff', float, .01))
        type_keys.append(('lr', 'learning_rate', float, 5e-4))
        type_keys.append(('gamma', 'gamma', float, 0.999))

        # RND
        type_keys.append(('s_clip', 's_clip', float, 10))
        type_keys.append(('r_clip', 'r_clip', float, 1))

        type_keys.append(('agent', 'agent', str, "ppo"))

        type_keys.append(('disable_wandb', 'disable_wandb', int, 1))

        # Should the agent's velocity be painted in the upper left corner of observations.
        # 1/0 means True/False
        # PAINT_VEL_INFO = -1 uses smart defaulting -- will default to 1 if GAME_TYPE is 'standard' (CoinRun), 0 otherwise
        type_keys.append(('pvi', 'paint_vel_info', int, -1, True))

        # Should batch normalization be used after each convolutional layer
        # 1/0 means True/False
        # This code only supports training-mode batch normalization (normalizing with statistics of the current batch).
        # In practice, we found this is nearly as effective as tracking the moving average of the statistics.
        # NOTE: Only applies to IMPALA and IMPALA-Large architectures
        type_keys.append(('norm', 'use_batch_norm', int, 0, True))

        # What dropout probability to use
        type_keys.append(('dropout', 'dropout', float, 0.0, True))

        # Use OPENAI version of dropout (SNI with lambda=1.0)
        bool_keys.append(('openai', 'openai', True))

        # # What dropout probability to use
        # type_keys.append(('dropout-openai', 'dropout_openai', float, 0.0, True))

        # Should data augmentation be used
        # 1/0 means True/False
        type_keys.append(('uda', 'use_data_augmentation', int, 0))

        # The l2 penalty to use during training
        type_keys.append(('l2', 'l2_weight', float, 0.0))

        # The probability the agent's action is replaced with a random action
        type_keys.append(('eps', 'epsilon_greedy', float, 0.0))

        # The number of frames to stack for each observation.
        # No frame stack is necessary if PAINT_VEL_INFO = 1
        type_keys.append(('fs', 'frame_stack', int, 1, True))

        # Should observations be transformed to grayscale
        # 1/0 means True/False
        type_keys.append(('ubw', 'use_black_white', int, 0, True))

        # Overwrite the latest save file after this many updates
        type_keys.append(('si', 'save_interval', int, 10))

        # The number of evaluation environments to use
        type_keys.append(('num-eval', 'num_eval', int, 20, False))

        # The number of episodes to evaluate with each evaluation environment
        type_keys.append(('rep', 'rep', int, 1))

        # Should half the workers act solely has test workers for evaluation
        # These workers will run on test levels and not contributing to training
        bool_keys.append(('test', 'test'))

        # Perform evaluation with all levels sampled from the training set
        bool_keys.append(('train-eval', 'train_eval'))

        # Perform evaluation with all levels sampled from the test set (unseen levels of high difficulty)
        bool_keys.append(('test-eval', 'test_eval'))

        # Only generate high difficulty levels
        bool_keys.append(('highd', 'high_difficulty'))

        # Use high resolution images for rendering
        bool_keys.append(('hres', 'is_high_res'))

        self.RES_KEYS = []

        for tk in type_keys:
            arg_keys.append(self.process_field(tk[1]))

            if (len(tk) > 4) and tk[4]:
                self.RES_KEYS.append(tk[1])

        for bk in bool_keys:
            arg_keys.append(bk[1])

            if (len(bk) > 2) and bk[2]:
                self.RES_KEYS.append(bk[1])

        self.arg_keys = arg_keys
        self.bool_keys = bool_keys
        self.type_keys = type_keys

        self.load_data = {}
        self.args_dict = {}

        # print("Rank {} is a test rank: {}".format(MPI.COMM_WORLD.Get_rank(), self.is_test_rank()))

    def is_test_rank(self):
        if self.TEST:
            rank = MPI.COMM_WORLD.Get_rank()
            return rank % self.TEST_RATIO == 1

        return False

    def get_test_frac(self):
        return .5 if self.TEST else 0

    def get_load_data(self, load_key='default'):
        if not load_key in self.load_data:
            return None

        return self.load_data[load_key]

    def set_load_data(self, ld, load_key='default'):
        self.load_data[load_key] = ld

    # The two methods below are modified. Why did they do these stealthy conversions to begin with?
    def process_field(self, name):
        # return name.replace('-','_')
        return name

    def deprocess_field(self, name):
        # return name.replace('_','-')
        return name

    def parse_all_args(self, args):
        assert isinstance(args, argparse.Namespace), 'expected argparse.Namespace object'
        update_dict = vars(args)
        self.parse_args_dict(update_dict)

    def parse_args_dict(self, update_dict):
        self.args_dict.update(update_dict)

        for ak in self.args_dict:
            val = self.args_dict[ak]

            if isinstance(val, str):
                val = self.process_field(val)

            setattr(self, ak.upper(), val)

        self.compute_args_dependencies()

    def compute_args_dependencies(self):
        if self.is_test_rank():
            self.NUM_LEVELS = 0
            self.USE_DATA_AUGMENTATION = 0
            self.EPSILON_GREEDY = 0
            self.HIGH_DIFFICULTY = 1

        if self.PAINT_VEL_INFO < 0:
            if self.GAME_TYPE == 'standard':
                self.PAINT_VEL_INFO = 1
            else:
                self.PAINT_VEL_INFO = 0

        if self.TEST_EVAL:
            self.NUM_LEVELS = 0
            self.HIGH_DIFFICULTY = 1

        self.TRAIN_TEST_COMM = MPI.COMM_WORLD.Split(1 if self.is_test_rank() else 0, 0)

    def get_load_filename(self, base_name=None, restore_id=None):
        if restore_id is None:
            restore_id = Config.RESTORE_ID

        if restore_id is None:
            return None
        
        filename = Config.get_save_file_for_rank(0, self.process_field(restore_id), base_name=base_name)

        return filename

    def get_save_path(self, runid=None):
        return self.WORKDIR + self.get_save_file(runid)

    def get_save_file_for_rank(self, rank, runid=None, base_name=None):
        if runid is None:
            runid = self.RUN_ID

        extra = ''

        if base_name is not None:
            extra = '_' + base_name

        return 'sav_' + runid + extra + '_' + str(rank)

    def get_save_file(self, runid=None, base_name=None):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        return self.get_save_file_for_rank(rank, runid, base_name=base_name)

    def get_arg_text(self):
        arg_strs = []

        for key in self.args_dict:
            arg_strs.append(key + '=' + str(self.args_dict[key]))

        return arg_strs

    def get_args_dict(self):
        _args_dict = {}
        _args_dict.update(self.args_dict)

        return _args_dict
        
    def initialize_args(self, use_cmd_line_args=True, **kwargs):
        default_args = {}

        for tk in self.type_keys:
            default_args[self.process_field(tk[1])] = tk[3]

        for bk in self.bool_keys:
            default_args[bk[1]] = False
        default_args.update(kwargs)
        # print("Default args: {}".format(default_args))

        parser = argparse.ArgumentParser()

        for tk in self.type_keys:
            parser.add_argument('-' + tk[0], '--' + self.deprocess_field(tk[1]), type=tk[2], default=default_args[tk[1]])

        for bk in self.bool_keys:
            parser.add_argument('--' + bk[0], dest=bk[1], action='store_true')
            bk_kwargs = {bk[1]: default_args[bk[1]]}
            parser.set_defaults(**bk_kwargs)

        if use_cmd_line_args:
            args = parser.parse_args()
        else:
            args = parser.parse_args(args=[])

        self.parse_all_args(args)

        return args

Config = ConfigSingle()

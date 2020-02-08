#!/usr/bin/env python3
import numpy as np

from baselines import bench
import os.path as osp
import os
from warnings import warn



from baselines.common.cmd_util import arg_parser

def arg_parser_common():
    import ast,json
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Walker2d-v2')
    parser.add_argument('--isatari', default=False, action='store_true')

    # parser.add_argument('--env', help='environment ID', type=str, default='AtlantisNoFrameskip')#TODO: tmp
    # parser.add_argument('--isatari', default=True, action='store_true')

    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--alg', help='', default='trgppo', type=str)#

    parser.add_argument('--cliptype', default='' , type=str)#wasserstein_wassersteinrollback_constant,kl_klrollback_constant_withratio
    parser.add_argument('--clipargs', default=dict(), type=json.loads)


    # The priority of the default args is defined by the order it appears.
    # The input args has highest priority
    # The cliptype has highest priority
    args_default_all = \
        dict(__envtype=dict(
            mujoco=dict(
                policy_type = 'MlpPolicyExt',
                n_steps = 1024,
                n_envs = 2,
                n_minibatches = 32,
                n_opt_epochs = 10,
                lr = 3e-4,
                coef_entropy = 0,
                eval_interval = 1,
                num_timesteps = int(1e6),
                save_interval = 10,
                logstd = 0,
                __env = dict(
                    Humanoid = dict(
                        n_envs = 64,
                        n_minibatches = 64,
                        num_timesteps = int(20e6),
                    ),
                    HalfCheetah = dict(
                        logstd = -1.34
                    )
                ),
                __cliptype = dict(
                    ratio=dict(clipargs=dict(cliprange=0.2)),
                    ratio_rollback=dict(
                        clipargs=dict(cliprange=0.2, slope_rollback=-0.3),
                        __env = dict(
                            Humanoid=dict(
                                logstd=-1.34657,
                                clipargs=dict(cliprange=0.2, slope_rollback=-0.02)
                            )
                        )
                    ),
                    ratio_strict=dict(clipargs=dict(cliprange=0.2)),
                    ratio_rollback_constant=dict(clipargs=dict(cliprange=0.2, slope_rollback=-0.3)),

                    a2c=dict(clipargs=dict(cliprange=0.1)),

                    kl=dict(
                        clipargs=dict(klrange=0.035, cliprange=0.2),
                        __env=dict(
                            Humanoid=dict(
                                logstd=-0.5,
                                clipargs=dict(klrange=0.05, cliprange=0.2)
                            )
                        )
                    ),
                    kl_strict=dict(clipargs=dict(klrange=0.025, cliprange=0.2)),
                    kl_ratiorollback=dict(clipargs=dict(klrange=0.03, slope_rollback=-0.05, cliprange=0.2)),
                    kl_klrollback_constant=dict(clipargs=dict(klrange=0.03, slope_rollback=-0.4, cliprange=0.2)),
                    kl_klrollback_constant_withratio=dict(
                        # The common args
                        clipargs=dict(klrange=0.03, slope_rollback=-5, slope_likelihood=1, cliprange=0.2),
                        # The args for specific env
                        __env=dict(
                            Humanoid=dict(
                                logstd=-0.5,
                                clipargs=dict(klrange=0.05, slope_rollback=-0.4, slope_likelihood=0, cliprange=0.2)
                            )
                        )
                    ),
                    kl_klrollback=dict(clipargs=dict(klrange=0.03, slope_rollback=-0.1, cliprange=0.2)),

                    # klrange is used for kl2clip, which could be None. If it's None, it is adjusted by cliprange.
                    # cliprange is used for value clip, which could be None. If it's None, it is adjusted by klrange.
                    kl2clip=dict(
                        clipargs=dict(klrange=None, adjusttype='base_clip_upper', cliprange=0.2, kl2clip_opttype='tabular',
                                      adaptive_range=''),
                        __env=dict(
                            Humanoid=dict(
                                logstd=-1.34657359,
                                clipargs=dict(klrange=0.03, slope_rollback=-5, slope_likelihood=1.)
                            )
                        )
                    ),
                    kl2clip_rollback=dict(
                        clipargs=dict(klrange=None, adjusttype='base_clip_upper', cliprange=0.2, kl2clip_opttype='tabular',
                                      adaptive_range='', slope_rollback=-0.3)
                    ),

                    adaptivekl=dict(clipargs=dict(klrange=0.01, cliprange=0.2)),
                    adaptiverange_advantage=dict(clipargs=dict(cliprange_min=0, cliprange_max=0.4, cliprange=0.2)),

                    wasserstein=dict(clipargs=dict(range=0.05, cliprange=0.2)),
                    wasserstein_rollback_constant=dict(clipargs=dict(range=0.05, slope_rollback=-0.4, cliprange=0.2)),
                )
            ),
            atari=dict(
                policy_type='CnnPolicy',
                n_steps = 128 ,
                n_envs = 8,
                n_minibatches = 4,
                n_opt_epochs = 4,
                lr = 2.5e-4,
                coef_entropy= 0.01,
                eval_interval=0,
                num_timesteps=int(1e7),
                save_interval = 400,
                logstd = 0,
                __cliptype= dict(
                    ratio=dict(clipargs=dict(cliprange=0.1)),
                    ratio_rollback=dict(clipargs=dict(cliprange=0.1, slope_rollback=-0.01)),

                    a2c=dict(clipargs=dict(cliprange=0.1)),

                    kl=dict(
                        clipargs=dict(klrange=0.001, cliprange=0.1, decay_threshold=0.),
                        # coef_entropy = 0 #TODO
                    ),
                    kl_ratiorollback=dict(clipargs=dict(klrange=0.001,slope_rollback=-0.05, cliprange=0.1, decay_threshold=0.)),
                    kl_klrollback_constant=dict(clipargs=dict(klrange=0.001, slope_rollback=-0.05, cliprange=0.1, decay_threshold=0.)),
                    kl_klrollback_constant_withratio= dict(
                        clipargs = dict(klrange=0.0008, slope_rollback=-20, slope_likelihood=1, cliprange=0.1, decay_threshold=0.),
                        coef_entropy=0,
                    ),

                    totalvariation=dict(clipargs=dict(range=0.02, cliprange=0.1, decay_threshold=0.)),
                    totalvariation_rollback_constant=dict(
                        clipargs=dict(range=0.02, slope_rollback=-0.05, cliprange=0.1, decay_threshold=0.)
                    ),
                    kl2clip=dict(
                        clipargs=dict(klrange=0.001, cliprange=0.1, kl2clip_opttype='tabular', adaptive_range=''),
                        coef_entropy=0
                    ),
                    adaptivekl=dict(
                        clipargs=dict(klrange=0.01, cliprange=0.1)
                    ),
                )
            )
        ))

    parser.add_argument('--lam', default=0.95, type=float )
    parser.add_argument('--lr', default=None, type=float)

    parser.add_argument('--policy_type', default=None, type=str)

    parser.add_argument('--log_dir_mode', default='finish_then_exit_else_overwrite', type=str)#overwrite,finish_then_exit_else_overwrite
    parser.add_argument('--name_group', default=None, type=str)
    parser.add_argument('--keys_group', default=[], type=ast.literal_eval)

    # architecture of network
    parser.add_argument('--policy_variance_state_dependent', default=False, type=ast.literal_eval)
    parser.add_argument('--hidden_sizes', default=64, type=ast.literal_eval)
    parser.add_argument('--num_layers', default=2, type=ast.literal_eval)
    parser.add_argument('--num_sharing_layers', default=0, type=int)
    parser.add_argument('--ac_fn', default='tanh', type=str)

    parser.add_argument('--coef_predict_task', default=0, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--lam_decay', default=False, type=ast.literal_eval)
    # ----- Please keep the default values of the following args to be None, the default value are different for different tasks
    parser.add_argument('--coef_entropy', default=None, type=float)
    parser.add_argument('--n_envs', default=None, type=int)
    parser.add_argument('--n_steps', default=None, type=int)
    parser.add_argument('--n_minibatches', default=None, type=int)
    parser.add_argument('--n_opt_epochs', default=None, type=int)
    parser.add_argument('--logstd', default=None, type=float)

    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--n_eval_epsiodes', default=1, type=int)
    parser.add_argument('--num_timesteps', type=int, default=None)
    parser.add_argument('--eval_interval', type=int, default=None)
    parser.add_argument('--save_interval', default=None, type=int)
    parser.add_argument('--save_debug', default=False, action='store_true')
    # parser.add_argument('--debug_halfcheetah', default=0, type=int)
    parser.add_argument('--is_multiprocess', default=0, type=ast.literal_eval)
    return parser, args_default_all


from toolsm import tools
from toolsm import logger as tools_logger
from baselines.ppo2_AdaClip.algs import *
def main():
    parser, args_default = arg_parser_common()
    args = parser.parse_args()


    import json
    from dotmap import DotMap
    from copy import copy, deepcopy
    keys_exclude = [ 'coef_predict_task', 'is_multiprocess', 'n_envs', 'eval_interval', 'n_steps', 'n_minibatches',
        'play', 'n_eval_epsiodes', 'force_write', 'kl2clip_sharelogstd','policy_variance_state_dependent',
                   'kl2clip_clip_clipratio', 'kl2clip_decay', 'lr', 'num_timesteps', 'gradient_rectify', 'rectify_scale','kl2clip_clipcontroltype', 'reward_scale', 'coef_predict_task','explore_additive_rate','explore_additive_threshold','explore_timesteps', 'debug_halfcheetah', 'name_project', 'n_opt_epochs', 'coef_entropy', 'log_interval', 'save_interval', 'save_debug', 'isatari', 'env_full', 'envtype']
    keys_exclude.extend(['logstd','lam','hidden_sizes','num_layers','num_sharing_layers','ac_fn','lam_decay','policy_type'])
    # TODO: These args should not be used as name of dir only if they are specified.
    # TODO: Split args into..... group_keys and run_keys.

    #  -------------------- prepare args

    args.env_full = args.env
    args.env = args.env_full.split('-v')[0]

    if not args.isatari:
        args.envtype = MUJOCO
        if '-v' not in args.env_full:
            args.env_full = f'{args.env}-v2'
    else:
        keys_exclude.append('logstd')
        args.envtype = ATARI
        # if 'NoFrameskip' not in args.env:
        #     args.env = f''
        if '-v' not in args.env_full:
            args.env_full = f'{args.env}-v4'
    tools.warn_(f'Run with setting for {args.envtype} task!!!!!')

    assert bool(args.alg) != bool(args.cliptype), 'Either alg or cliptype should be specified'
    if args.alg: # For release
        args.cliptype = alg2cliptype[args.alg]
        keys_exclude.append('cliptype')
        if len(args.keys_group) ==0:
            args.keys_group = ['alg']
        if args.name_group is None:
            args.name_group = ''
    else: # For debug
        keys_exclude.append('alg')
        if len(args.keys_group) ==0:
            args.keys_group = ['cliptype','clipargs']
        if args.name_group is None:
            args.name_group = 'tmp'


    # ------ Set the values of args
    def update_dict(dictmain, dictnew):
        for key_arg in dictnew:
            if key_arg.startswith('__'):
                # This means that the value are customized for the specific values
                key_interest  = key_arg[2:] #e.g., __cliptype
                value_interest  = dictmain[key_interest] #Search value from dictmain. e.g., kl_klrollback_constant_withratio
                if value_interest in dictnew[ key_arg ].keys():
                    dictmain = update_dict( dictmain, dictnew[ key_arg ][value_interest])
            else:
                if isinstance(dictnew[key_arg], dict) and key_arg in dictmain.keys():
                    dictmain[key_arg].update( dictnew[key_arg] )
                else:
                    dictmain[key_arg] = copy( dictnew[key_arg])
        return dictmain

    def reform_specific_dict(d):
        dictmain = dict( (k,v) for k,v in d.items() if not k.startswith('__') )
        dictspecific = dict( (k,v) for k,v in d.items() if k.startswith('__') )
        return update_dict( dictmain, dictspecific )


    # If the value of the following args are None, then it is setted by the following values
    keys_del = []
    args = vars(args)
    keys = list(args.keys())
    for key in keys:
        if args[key] is None:
            del args[key] #Delete the value of args
            keys_del.append( key )
    if len(keys_del) > 0:
        print( 'The following args are not provided value by the args. They will used built-in values.\n', ', '.join(keys_del) )

    # args__ = update_dict( copy(args_default), args ) # We need to update the basic args, e.g., env, cliptype
    # args__  = reform_specific_dict( args__)
    # The following operations may seems strange. Maybe I will give a more clear one in the furture.
    args__ = update_dict( deepcopy(args), args_default ) # generate the default value from args_default
    args = update_dict( args__, args ) # The priority of the customed value is highest
    for key in keys_del: # make sure that keys_del are within args.keys()
        assert key in args.keys(), key
    # print( json.dumps(args, indent=True) )
    # exit()
    # TODO prepare_dir: change .finish_indicator to finishi_indictator, which is more clear.
    # --- prepare dir
    import baselines
    root_dir = tools_logger.get_logger_dir(  'baselines', 'results', baselines )
    args = tools_logger.prepare_dirs( args, key_first='env', keys_exclude=keys_exclude, dirs_type=['log' ], root_dir=root_dir )
    # --- prepare args for use
    args.cliptype = ClipType[ args.cliptype ]

    args.zip_dirs = ['model','monitor']
    for d in args.zip_dirs:
        args[f'{d}_dir'] = osp.join(args.log_dir, d)
        os.mkdir( args[f'{d}_dir'] )

    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2_AdaClip import ppo2
    # from baselines.ppo2_AdaClip import ppo2_kl2clip_conservative as ppo2
    import baselines.ppo2_AdaClip.policies as plcs
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


    set_global_seeds(args.seed)
    policy = getattr(plcs, args.policy_type)


    # ------ prepare env
    # args.eval_model = args.n_eval_epsiodes > 0
    if args.envtype == MUJOCO:
        def make_mujoco_env(rank=0):
            def _thunk():
                env = gym.make(args.env_full)
                env.seed(args.seed + rank)
                env = bench.Monitor(env, os.path.join(args.log_dir, 'monitor', str(rank)), allow_early_resets=True)
                return env

            return _thunk

        if args.n_envs == 1:
            env = DummyVecEnv([make_mujoco_env()])
        else:
            from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
            env = SubprocVecEnv([make_mujoco_env(i) for i in range(args.n_envs)])
        env = VecNormalize(env, reward_scale=args.reward_scale)

        env_test = None
        if args.n_eval_epsiodes > 0:
            if args.n_eval_epsiodes == 1:
                env_test = DummyVecEnv([make_mujoco_env()])
            else:
                from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
                env_test = SubprocVecEnv([make_mujoco_env(i) for i in range(args.n_eval_epsiodes)])
            env_test = VecNormalize(env_test, ret=False, update=False)  # It doesn't need to normalize return
    else:
        from baselines.common.vec_env.vec_frame_stack import VecFrameStack
        from baselines.common.cmd_util import make_atari_env
        env = VecFrameStack(make_atari_env(args.env_full, num_env=args.n_envs, seed=args.seed), 4)
        env_test = None
        #  TODO : debug VecFrame
        if args.n_eval_epsiodes > 0:
            env_test = VecFrameStack(make_atari_env(args.env_full, num_env=args.n_eval_epsiodes, seed=args.seed), 4)
            # env_test.reset()
            # env_test.render()
    # ----------- learn
    if args.envtype == MUJOCO:
        lr = args.lr
        # cliprange = args.clipargs.cliprange
    elif args.envtype == ATARI:
        lr = lambda f: f * args.lr
        # cliprange = lambda f: f*args.clipargs.cliprange if args.clipargs.cliprange is not None else None
    # print('action_space',env.action_space)
    ppo2.learn(policy=policy, env=env, env_eval=env_test, n_steps=args.n_steps, nminibatches=args.n_minibatches,
               lam=args.lam, gamma=0.99, n_opt_epochs=args.n_opt_epochs, log_interval=args.log_interval,
               ent_coef=args.coef_entropy,
               lr=lr,
               total_timesteps=args.num_timesteps,
               cliptype=args.cliptype, save_interval=args.save_interval, args=args)

    tools_logger.finish_dir( args.log_dir )


if __name__ == '__main__':
    main()
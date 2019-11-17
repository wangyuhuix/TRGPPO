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
    parser.add_argument('--env', help='environment ID', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--is_atari', default=False, action='store_true')

    # parser.add_argument('--env', help='environment ID', type=str, default='PongNoFrameskip')
    # parser.add_argument('--is_atari', default=True, action='store_true')

    parser.add_argument('--seed', help='RNG seed', type=int, default=0)


    parser.add_argument('--cliptype', default='kl2clip', type=str)#wasserstein_wassersteinrollback_constant
    # parser.add_argument('--cliprange', default=0.2, type=float)
    # import demjson
    parser.add_argument('--clipargs', default=dict(), type=json.loads)

    clipargs_default_all = {
        MUJOCO: dict(
            ratio=dict(cliprange=0.2),
            ratio_rollback=dict(cliprange=0.2, slope_rollback=-0.3),
            ratio_strict=dict(cliprange=0.2),
            ratio_rollback_constant=dict(cliprange=0.2, slope_rollback=-0.3),


            a2c=dict(cliprange=0.1),

            wasserstein = dict(range=0.05, cliprange=0.2),
            wasserstein_wassersteinrollback_constant = dict(range=0.05, slope_rollback=-0.4, cliprange=0.2),

            kl=dict(klrange=0.03, cliprange=0.2),
            kl_strict=dict(klrange=0.025, cliprange=0.2),
            kl_ratiorollback=dict(klrange=0.03, slope_rollback=-0.05, cliprange=0.2),
            kl_klrollback_constant=dict(klrange=0.03, slope_rollback=-0.1, cliprange=0.2),
            kl_klrollback=dict(klrange=0.03, slope_rollback=-0.1, cliprange=0.2),

            # base_clip_lower, base_clip_upper
            # kl2clip = dict(klrange=0.03, adjusttype='origin', cliprange=0.2, kl2clip_opttype='tabular', adaptive_range=''),
            kl2clip=dict(klrange=None, adjusttype='base_clip_upper', cliprange=0.2, kl2clip_opttype='tabular', adaptive_range=''),
            kl2clip_rollback=dict(klrange=None, adjusttype='base_clip_upper', cliprange=0.2, kl2clip_opttype='tabular', adaptive_range='', slope_rollback=-0.3),

            # kl2clip = dict( klrange=None, adjusttype='base_clip_lower', cliprange=0.2)
            # kl2clip=dict(klrange=None, adjusttype='base_clip_upper', cliprange=0.2, kl2clip_opttype='tabular'),#nn
            # klrange is used for kl2clip, which could be None. If it's None, it is adjusted by cliprange.
            # cliprange is used for value clip, which could be None. If it's None, it is adjusted by klrange.
            adaptivekl=dict(klrange=0.01, cliprange=0.2),
            adaptiverange_advantage = dict(cliprange_min=0,cliprange_max=0.4,cliprange=0.2)
        ),
        ATARI: dict(
            # TODO:!!!  Please modify the parameters here
            ratio=dict(cliprange=0.1),
            ratio_rollback=dict(cliprange=0.1, slope_rollback=-0.3),
            ratio_strict=dict(cliprange=0.1),
            ratio_rollback_constant=dict(cliprange=0.1, slope_rollback=-0.3),

            a2c=dict(cliprange=0.1),

            kl=dict(klrange=0.03, cliprange=0.1),
            kl_strict=dict(klrange=0.025, cliprange=0.1),
            kl_ratiorollback=dict(klrange=0.03, slope_rollback=-0.05, cliprange=0.1),
            kl_klrollback_constant=dict(klrange=0.03, slope_rollback=-0.1, cliprange=0.1),
            kl_klrollback=dict(klrange=0.03, slope_rollback=-0.1, cliprange=0.1),

            # kl2clip = dict(klrange=0.03, adjusttype='origin', cliprange=0.2)
            # kl2clip = dict( klrange=None, adjusttype='base_clip_lower', cliprange=0.2)
            kl2clip=dict(klrange=0.001, cliprange=0.1, kl2clip_opttype='tabular', adaptive_range=''),
            # klrange is used for kl2clip, which could be None. If it's None, it is adjusted by cliprange.
            # cliprange is used for value clip, which could be None. If it's None, it is adjusted by klrange.

            adaptivekl=dict(klrange=0.01, cliprange=0.1)
        )
    }
    # parser.add_argument('--cliptype', default='origin', type=str)
    # parser.add_argument('--slope', default=0, type=float)
    # parser.add_argument('--cliprange', default=0.2, type=ast.literal_eval)
    # parser.add_argument('--delta_kl', default=None, type=ast.literal_eval)

    parser.add_argument('--lam', default=0.95, type=float )
    parser.add_argument('--lr', default=None, type=float)

    parser.add_argument('--policy_type', default=None, type=str)

    parser.add_argument('--log_dir_mode', default='finish_then_exit_else_overwrite', type=str)#overwrite,finish_then_exit_else_overwrite
    parser.add_argument('--name_group', default='tmp', type=str)
    parser.add_argument('--keys_group', default=['cliptype','clipargs'], type=ast.literal_eval)

    # architecture of network
    parser.add_argument('--policy_variance_state_dependent', default=False, type=ast.literal_eval)
    parser.add_argument('--hidden_sizes', default=64, type=ast.literal_eval)
    parser.add_argument('--num_layers', default=2, type=ast.literal_eval)
    parser.add_argument('--num_sharing_layers', default=0, type=int)
    parser.add_argument('--ac_fn', default='tanh', type=str)

    # parser.add_argument('--explore', default=0, type=int)
    # parser.add_argument('--explore_timesteps', default=0, type=int)
    # parser.add_argument('--explore_additive_threshold', default=None, type=float)
    # parser.add_argument('--explore_additive_rate', default=0, type=float)



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
    args_default_all = \
        {
            # MUJOCO
            MUJOCO: dict(
                policy_type = dict(_default='MlpPolicyExt'),
                n_steps = dict( _default=1024  ),
                n_envs = dict( Humanoid=64, _default=2 ),
                n_minibatches = dict( Humanoid=64, _default=32 ),
                n_opt_epochs = dict(_default=10),
                lr = dict(_default=3e-4),
                coef_entropy = dict( _default=0 ),
                eval_interval = dict( _default=1 ),
                num_timesteps = dict( Humanoid=int(20e6),  _default=int(1e6) ),
                save_interval = dict( _default=10 ),
                logstd = dict( HalfCheetah=-1.34, Humanoid=-1.34657, _default=0, ),
            ),
            # ATARI
            ATARI: dict(
                policy_type=dict(_default='CnnPolicy'),
                n_steps = dict( _default=128  ),
                n_envs = dict( _default=8 ),
                n_minibatches = dict( _default=4 ),
                n_opt_epochs = dict( _default=4 ),
                lr = dict(_default=2.5e-4 ),
                coef_entropy= dict(_default=0),#TODO: tmp for kl2clip
                eval_interval=dict(_default=-1),
                num_timesteps=dict(_default=int(1e7)),
                save_interval = dict(  _default=400 ),
            )
        }
    # parser.add_argument('--debug_halfcheetah', default=0, type=int)
    parser.add_argument('--is_multiprocess', default=0, type=ast.literal_eval)
    return parser, clipargs_default_all, args_default_all


from toolsm import tools
from toolsm import logger as tools_logger
from baselines.ppo2_AdaClip.algs import *
def main():
    parser, clipargs_default_all, args_default_all = arg_parser_common()
    args = parser.parse_args()


    import json
    from dotmap import DotMap
    keys_exclude = [ 'coef_predict_task', 'is_multiprocess', 'n_envs', 'eval_interval', 'n_steps', 'n_minibatches',
        'play', 'n_eval_epsiodes', 'force_write', 'kl2clip_sharelogstd','policy_variance_state_dependent',
                   'kl2clip_clip_clipratio', 'kl2clip_decay', 'lr', 'num_timesteps', 'gradient_rectify', 'rectify_scale','kl2clip_clipcontroltype', 'reward_scale', 'coef_predict_task','explore_additive_rate','explore_additive_threshold','explore_timesteps', 'debug_halfcheetah', 'name_project', 'env_pure', 'n_opt_epochs', 'coef_entropy', 'log_interval', 'save_interval', 'save_debug', 'is_atari']
    # 'is_atari'


    #  -------------------- prepare args


    args.env_pure = args.env.split('-v')[0]

    # env_mujocos = 'InvertedPendulum,InvertedDoublePendulum,HalfCheetah,Hopper,Walker2d,Ant,Reacher,Swimmer,Humanoid'
    # env_mujocos = tools.str2list(env_mujocos)
    if not args.is_atari:
        env_type = MUJOCO
        if '-v' not in args.env:
            args.env = f'{args.env}-v2'
    else:
        env_type = ATARI
        if '-v' not in args.env:
            args.env = f'{args.env}-v4'
    tools.warn_(f'Run with setting for {env_type} task!!!!!')

    # --- set value of clipargs
    clipargs_default = clipargs_default_all[env_type]

    clipargs = clipargs_default[ args.cliptype ].copy()
    clipargs.update( args.clipargs )
    args.clipargs = clipargs

    # --- prepare other args
    # If the value of the following args are None, then it is setted by the following values
    args_default = args_default_all[ env_type ]
    args = DotMap( vars( args ))
    print("The followng arg value is None, thus they are setted by built-in value:")

    for argname in args_default.keys():
        if args[argname] is None:
            if args.env_pure in args_default[argname].keys():
                args[argname] = args_default[argname][args.env_pure]
            else:
                args[argname] = args_default[argname]['_default']
            print(f"{argname}={args[argname]}")
    # print( json.dumps( args.toDict(), indent='\t') )
    # exit()
    # TODO prepare_dir: change .finish_indicator to finishi_indictator, which is more clear.
    # --- prepare dir
    import baselines
    root_dir = tools_logger.get_logger_dir(  'baselines', baselines, 'results' )
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
    if env_type == MUJOCO:
        def make_mujoco_env(rank=0):
            def _thunk():
                env = gym.make(args.env)
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
        env = VecFrameStack(make_atari_env(args.env, num_env=args.n_envs, seed=args.seed), 4)
        env_test = None
        #  TODO : debug VecFrame
        if args.n_eval_epsiodes > 0:
            env_test = VecFrameStack(make_atari_env(args.env, num_env=args.n_eval_epsiodes, seed=args.seed), 4)
            # env_test.reset()
            # env_test.render()
    # ----------- learn
    if env_type == MUJOCO:
        lr = args.lr
        # cliprange = args.clipargs.cliprange
    elif env_type == ATARI:
        lr = lambda f: f * args.lr
        # cliprange = lambda f: f*args.clipargs.cliprange if args.clipargs.cliprange is not None else None
    args.env_type = env_type
    ppo2.learn(policy=policy, env=env, env_eval=env_test, n_steps=args.n_steps, nminibatches=args.n_minibatches,
               lam=args.lam, gamma=0.99, n_opt_epochs=args.n_opt_epochs, log_interval=args.log_interval,
               ent_coef=args.coef_entropy,
               lr=lr,
               total_timesteps=args.num_timesteps,
               cliptype=args.cliptype, save_interval=args.save_interval, args=args)

    tools_logger.finish_dir( args.log_dir )


if __name__ == '__main__':
    main()

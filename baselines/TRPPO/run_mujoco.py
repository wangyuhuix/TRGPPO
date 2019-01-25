#!/usr/bin/env python3
import numpy as np

from baselines import bench, logger
import os.path as osp
import os
from warnings import warn


def train(env_id, clipped_type, num_timesteps, seed, args):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.TRPPO import ppo2
    import baselines.TRPPO.policies as plcs
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    if env_id == 'Humanoid-v2':
        from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
        def make_env(rank):
            def _thunk():
                env = gym.make(env_id)
                env.seed(seed + rank)
                env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
                return env

            return _thunk

        nenvs = 32
        env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
        env = VecNormalize(env)
    else:
        def make_env():
            env = gym.make(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env

        env = DummyVecEnv([make_env])
        env = VecNormalize(env)

    set_global_seeds(seed)

    policy = getattr(plcs, args.policy_type)
    if 'Lstm' in args.policy_type:
        nminibatches = 1
    else:
        nminibatches = 32
    if env_id == 'Humanoid-v2':
        model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=64,
                           lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                           ent_coef=0.0,
                           lr=args.lr,
                           total_timesteps=num_timesteps,
                           clipped_type=clipped_type, args=args)


    else:
        model = ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=nminibatches,
                           lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                           ent_coef=0.0,
                           lr=args.lr,
                           total_timesteps=num_timesteps,
                           clipped_type=clipped_type, args=args)
    return model, env


from baselines.common.cmd_util import arg_parser


def mujoco_arg_parser():
    import ast
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--clipped_type', default='kl2clip', type=str)
    parser.add_argument('--use_tabular', default=False, type=ast.literal_eval)
    parser.add_argument('--cliprange', default=0.2, type=ast.literal_eval)
    parser.add_argument('--delta_kl', default=None, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)

    # TODO: 修改根路径

    root_dir_default = '/tmp/baselines'
    if not os.path.exists(root_dir_default):
        tools.mkdir(root_dir_default)

    parser.add_argument('--root_dir', default=root_dir_default, type=str)
    parser.add_argument('--sub_dir', default=None, type=str)
    parser.add_argument('--policy_type', default='MlpPolicy', type=str)
    parser.add_argument('--force_write', default=1, type=int)
    return parser


from baselines.common import tools
import json


def main():
    args = mujoco_arg_parser().parse_args()
    if args.clipped_type == 'kl2clip':
        name_tmp = ''
        assert (args.cliprange is None) is not (
                args.delta_kl is None), "TRPPO can receive only one of cliprange and delta_kl arguments"
        if args.cliprange:
            args.kl2clip_clipcontroltype = 'base-clip'
        else:
            args.kl2clip_clipcontroltype = 'none-clip'
    else:
        name_tmp = ''
        assert args.cliprange, "PPO has to receive a cliprange parameter, the default one is 0.2"

    # --- Generate sub_dir of log dir and model dir
    split = ','
    if args.sub_dir is None:
        keys_except = ['env', 'play', 'root_dir', 'sub_dir', 'force_write', 'lr', 'kl2clip_clipcontroltype']
        # TODO: tmp for kl2clip_sharelogsigma
        keys_fmt = {'num_timesteps': '.0e'}
        args_dict = vars(args)
        sub_dir = args.env
        if not args.clipped_type in ['kl2clip']:
            keys_except += ['delta_kl']
        if not args.clipped_type in ['origin', 'kl2clip', 'a2c']:
            keys_except += ['cliprange']

        # --- add keys common
        for key in args_dict.keys():
            if key not in keys_except and key not in keys_fmt.keys():
                sub_dir += f'{split} {key}={args_dict[key]}'
        # --- add keys which has specific format
        for key in keys_fmt.keys():
            sub_dir += f'{split} {key}={args_dict[key]:{keys_fmt[key]}}'
        sub_dir += ('' if name_tmp == '' else f'{split} {name_tmp}')
        args.sub_dir = sub_dir

    tools.mkdir(f'{args.root_dir}/log')
    tools.mkdir(f'{args.root_dir}/model')
    args.log_dir = f'{args.root_dir}/log/{args.sub_dir}'
    args.model_dir = f'{args.root_dir}/model/{args.sub_dir}'
    force_write = args.force_write
    # Move Dirs
    if osp.exists(args.log_dir) or osp.exists(args.model_dir):  # modify name if exist
        print(
            f"Exsits directory! \n log_dir:'{args.log_dir}' \n model_dir:'{args.model_dir}'\nMove to discard(y or n)?",
            end='')
        if force_write > 0:
            cmd = 'y'
        elif force_write < 0:
            exit()
        else:
            cmd = input()
        if cmd == 'y':
            log_dir_new = args.log_dir.replace('/log/', '/log_discard/')
            model_dir_new = args.model_dir.replace('/model/', '/model_discard/')
            import itertools
            if osp.exists(log_dir_new) or osp.exists(model_dir_new):
                for i in itertools.count():
                    suffix = f' {split} {i}'
                    log_dir_new = f'{args.root_dir}/log_discard/{args.sub_dir}{suffix}'
                    model_dir_new = f'{args.root_dir}/model_discard/{args.sub_dir}{suffix}'
                    if not osp.exists(log_dir_new) and not osp.exists(model_dir_new):
                        break
            print(f"Move log_dir '{args.log_dir}' \n   to '{log_dir_new}'. \n"
                  f"Move model_dir '{args.model_dir}' \n to '{model_dir_new}'"
                  f"\nConfirm move(y or n)?", end='')
            if force_write > 0:
                cmd = 'y'
            elif force_write < 0:
                exit()
            else:
                cmd = input()
            if cmd == 'y':
                import shutil
                if osp.exists(args.log_dir):
                    shutil.move(args.log_dir, log_dir_new)
                if osp.exists(args.model_dir):
                    shutil.move(args.model_dir, model_dir_new)
            else:
                print("Please Rename 'name_tmp'")
                exit()
        else:
            print("Please Rename 'name_tmp'")
            exit()

    os.mkdir(args.log_dir)
    os.mkdir(args.model_dir)
    # exit()

    os.mkdir(osp.join(args.model_dir, 'cliprange_max'))
    os.mkdir(osp.join(args.model_dir, 'cliprange_min'))
    os.mkdir(osp.join(args.model_dir, 'actions'))
    os.mkdir(osp.join(args.model_dir, 'mu0_logsigma0'))
    os.mkdir(osp.join(args.model_dir, 'kls, ratios'))
    os.mkdir(osp.join(args.model_dir, 'advs'))

    args_str = vars(args)
    with open(f'{args.log_dir}/args.json', 'w') as f:
        json.dump(args_str, f, indent=4, separators=(',', ':'))
    logger.configure(args.log_dir)
    model, env = train(env_id=args.env, clipped_type=args.clipped_type, num_timesteps=args.num_timesteps,
                       seed=args.seed, args=args)
    # model, env = train(args.env, num_timesteps=10, seed=args.seed)

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    main()

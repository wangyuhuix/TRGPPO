#!/usr/bin/env python3
import ast
import os
import sys
import os.path as osp
from baselines.common import tools

from baselines import logger
from baselines.common.cmd_util import make_atari_env, arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.TRPPO import ppo2
from baselines.TRPPO.policies import CnnPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf


def train(env_id, clipped_type, num_timesteps, seed, args, policy):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    env = VecFrameStack(make_atari_env(env_id, num_env=8, seed=seed), 4)  # TODO: 注意是8个进程
    policy = {'cnn': CnnPolicy, 'mlp': MlpPolicy}[policy]
    ent_coef = 0.01 if args.clipped_type == 'origin' else 0
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
               lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
               ent_coef=ent_coef,
               lr=lambda f: f * 2.5e-4,
               total_timesteps=int(num_timesteps * 1.1),
               clipped_type=clipped_type, args=args,
               save_interval=200,
               )

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--clipped_type', default='kl2clip', type=str)
    parser.add_argument('--use_tabular', default=False, type=ast.literal_eval)
    parser.add_argument('--cliprange', default=0.1, type=ast.literal_eval)
    parser.add_argument('--delta_kl', default=0.001, type=float)
    root_dir_default = '/tmp/baselines'
    if not os.path.exists(root_dir_default):
        tools.mkdir(root_dir_default)

    parser.add_argument('--root_dir', default=root_dir_default, type=str)
    parser.add_argument('--sub_dir', default=None, type=str)
    parser.add_argument('--force_write', default=1, type=int)
    return parser

from baselines.common import tools
import json

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    args = parser.parse_args()
    if args.clipped_type == 'kl2clip':
        name_tmp = ''
        if args.cliprange and 'NoFrameskip-v4' not in args.env:
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
    # os.mkdir(osp.join(args.model_dir, 'mu0_logsigma0'))
    os.mkdir(osp.join(args.model_dir, 'kls, ratios'))
    os.mkdir(osp.join(args.model_dir, 'advs'))

    args_str = vars(args)
    with open(f'{args.log_dir}/args.json', 'w') as f:
        json.dump(args_str, f, indent=4, separators=(',', ':'))

    logger.configure(args.log_dir)
    train(env_id=args.env, clipped_type=args.clipped_type, num_timesteps=args.num_timesteps,
          seed=args.seed, args=args, policy=args.policy)


if __name__ == '__main__':
    main()

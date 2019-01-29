# Trust Region-Guided Proximal Policy Optimization

Tensorflow implementation of Trust Region-Guided Proximal Policy Optmization (TRPPO). The original code was forked from [OpenAI baselines](https://github.com/openai/baselines).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks and [Atari](https://www.atari.com/) discrete game tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [tensorflow1.10](https://www.tensorflow.org/) and Python 3.6.

## Usage

### Command Line arguments

* env: environment ID
* seed: random seed
* num_timesteps: number of timesteps
* play: whether to play the task after the training process
* clipped_type: 
  * origin: for PPO
  * kl2clip: for our proposed TRPPO
* use_tabular: whether to use a table to save the clipping range in order to accelerate the calculating clipping range process.
* cliprange: clipping range. for mujoco the default is 0.2, for atari the default is 0.1
* delta_kl: default is None. Not if the delta_kl is given, the TRPPO is gonna to use TRPPO-$\delta$ version. Otherwise it will be TRPPO-$\epsilon$ version. And 
* lr: learning rate. default is 3e-4(for adam).

### Continuous Task

TRPPO-$\epsilon$ on a single environment can be run by calling:

```shell
python -m baselines.TRPPO.run_mujoco --env=Walker2d-v2 --clipped_type=kl2clip --cliprange=0.2 --seed=0
```

TRPPO-$\delta$ on a single environment can be run by calling:

```shell
python -m baselines.TRPPO.run_mujoco --env=Swimmer-v2 --clipped_type=kl2clip --delta_kl=0.03 --cliprange=None --seed=1
```

> * Algorithms which TRPPO compares against (ACKTR, PPO) can be found at [OpenAI baselines repository](https://github.com/openai/baselines). But when change clipped_type in our implementation to origin, it will change to PPO algorithm.
>
> * We also provide a tabular version to calculate clipping ranges. To run it, just add --use_tabular=True.

### Discrete Task

TRPPO-$\delta$ on single environments can be run by calling:

```shell
python -m baselines.TRPPO.run_atari --env=BreakoutNoFrameskip-v4  --seed=0 --clipped_type=kl2clip --delta_kl=0.001 --use_tabular=True
```

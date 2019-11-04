# Trust Region-Guided Proximal Policy Optimization

Source code for the paper: [Trust Region-Guided Proximal Policy Optmization (TRPPO)](https://arxiv.org/abs/1901.10314). The original code was forked from [OpenAI baselines](https://github.com/openai/baselines).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks and [Atari](https://www.atari.com/) discrete game tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [tensorflow1.10](https://www.tensorflow.org/) and Python 3.6.

## Usage

### Command Line arguments

* env: environment ID
* seed: random seed
* num_timesteps: number of timesteps

### Continuous Task

TRGPPO can be run by calling:

```shell
python -m baselines.ppo2_AdaClip.run --env=Walker2d-v2 --seed=0
```

### Discrete Task

To be coming soon!
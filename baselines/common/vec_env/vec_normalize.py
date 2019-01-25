from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
import numpy as np

import joblib

class VecNormalize(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.variables_name_save = ['clipob','cliprew','ret','gamma', 'epsilon'  ]


    def load(self, path):
        vs = joblib.load(path)
        assert len(vs) == len( self.variables_name_save ) + 2
        for i,v_name in enumerate( self.variables_name_save ):
            setattr( self, v_name, vs[i] )
        self.ob_rms.load(  vs[-2]  )
        self.ret_rms.load( vs[-1] )

    def save(self, path):
        variables = []
        for v in self.variables_name_save:
            variables.append( getattr(self,v) )
        variables.append( self.ob_rms.save() )
        variables.append( self.ret_rms.save() )
        joblib.dump( variables, path )


    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)



if __name__ == '__main__':
    import gym
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    import numpy as np
    def make_env():
        env_id = 'InvertedPendulum-v2'
        seed = 1
        env = gym.make(env_id)
        # TODO: alternate random seed or not ?
        env.seed(seed)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize( env, clipob=29, epsilon=0.2 )
    env.reset()
    for t in range(100):
        env.step( env.action_space.sample() )

    env.save('../../t/a.env')
    # exit()


    env_new = DummyVecEnv([make_env])
    env_new = VecNormalize( env_new )
    env_new.load('../../t/a.env')

    for v_name in env.variables_name_save:
        assert getattr( env, v_name, ) == getattr( env_new, v_name ), v_name

    for v_name in env.ob_rms.variables_name_save:
        v1 = getattr( env.ob_rms, v_name )
        v2 = getattr( env_new.ob_rms, v_name )

        if isinstance( v1, np.ndarray ):
            assert  (v1 == v2).all()
        else:
            assert  (v1==v2)


    exit()

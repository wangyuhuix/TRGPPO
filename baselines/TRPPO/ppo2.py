import os
import os.path as osp
import time
from collections import deque

import joblib
import numpy as np
import tensorflow as tf
import gym
from baselines import logger
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_normalize import VecNormalize

save_env = None
load_env = None


def set_save_load_env(env):
    global save_env, load_env
    save_env = save_env_fn(env)
    load_env = load_env_fn(env)


def save_env_fn(env):
    def save_env(save_path):
        if isinstance(env, VecNormalize):
            env.save(save_path + '.env')

    return save_env


def load_env_fn(env):
    def load_env(load_path):
        if isinstance(env, VecNormalize):
            env.load(load_path + '.env')

    return load_env


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, clipped_type, args):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        # Add by xiaoming
        CLIPRANGE_min = tf.placeholder(tf.float32, [None])
        CLIPRANGE_max = tf.placeholder(tf.float32, [None])

        neglogpac = train_model.pd.neglogp(A)
        entropy_vector = train_model.pd.entropy()
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = tf.maximum(vf_losses1, vf_losses2)
        vf_loss = .5 * tf.reduce_mean(vf_loss)

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        pg_loss = None

        new_pd = train_model.pd
        if hasattr(new_pd, 'flat'):
            flat_shape = new_pd.flat.shape
        else:
            flat_shape = new_pd.logits.shape

        old_policyflat = tf.placeholder(tf.float32, shape=flat_shape, name='old_policyflat')
        old_pd = train_model.pdtype.pdfromflat(old_policyflat)
        kl = old_pd.kl(new_pd)
        if clipped_type == ClipType.kl2clip:
            pg_losses = -ADV * ratio
            # pg_losses2 = -ADV * tf.clip_by_value(ratio, tf.maximum(CLIPRANGE_min, 0.5), tf.minimum(CLIPRANGE_max, 1.5))
            pg_losses2 = -ADV * tf.clip_by_value(ratio, CLIPRANGE_min, CLIPRANGE_max)
            policyobj_vector = - tf.maximum(pg_losses, pg_losses2)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            clipfrac = tf.reduce_mean(
                tf.to_float(tf.where(ADV >= 0, tf.greater(ratio, CLIPRANGE_max), tf.less(ratio, CLIPRANGE_min))))
        elif clipped_type == ClipType.origin:
            # original clipped version
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        elif clipped_type == ClipType.origin_strict:
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(pg_losses2)
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        elif clipped_type == ClipType.a2c:
            pg_losses = -ADV * ratio
            pg_loss = tf.reduce_mean(pg_losses)
            clipfrac = tf.constant(0)
        else:
            raise Exception('No such clip type.')

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, obs, returns, masks, actions, values, neglogpacs, advs, cliprange,
                  cliprange_min=None, cliprange_max=None, states=None, policyflats=None):
            assert policyflats is not None
            if clipped_type == ClipType.kl2clip:
                assert cliprange_min is not None and cliprange_max is not None
                td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                          CLIPRANGE: cliprange, CLIPRANGE_min: cliprange_min, CLIPRANGE_max: cliprange_max,
                          OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, old_policyflat: policyflats}
            else:
                td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                          CLIPRANGE: cliprange,
                          OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, old_policyflat: policyflats}

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks  # for LstmPolicy
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, kl, ratio, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def create_summary_writer(graph_dir: str) -> tf.summary.FileWriter:
            # save graph to disk
            if tf.gfile.Exists(graph_dir):
                pass
                # tf.gfile.DeleteRecursively(graph_dir)
            return tf.summary.FileWriter(graph_dir, sess.graph)

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)
            save_env(save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            load_env(load_path)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        # self.step = act_model.step
        self.step = act_model.step_policyflat
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.create_summary_writer = create_summary_writer
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101


from baselines.common.tools import save_vars


class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_policyflats = [], [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs, policyflats = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_policyflats.append(policyflats)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        # save_vars('t/a.pkl',mb_policyflats)
        # exit()
        mb_policyflats = np.asarray(mb_policyflats, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_policyflats)),
                mb_states, epinfos)


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val

    return f


from enum import Enum


class ClipType(Enum):
    origin = 1
    kl2clip = 2
    a2c = 3
    judgekl = 4
    origin_strict = 5
    conservative = 6


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=1, nminibatches=4, noptepochs=4,
          save_interval=10, load_path=None, clipped_type, args=None):
    clipped_type = ClipType[clipped_type]
    print(f'Logger.CURRENT.dir is {logger.Logger.CURRENT.dir}')
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    set_save_load_env(env)
    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm, clipped_type=clipped_type, args=args)
    if save_interval and args.model_dir:
        import cloudpickle
        with open(osp.join(args.model_dir, 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    writer = model.create_summary_writer(args.log_dir)

    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    # epinfobuf = deque(maxlen=3)
    tfirststart = time.time()

    # assert len(ac_space.shape) == 1
    if clipped_type == ClipType.kl2clip:
        if isinstance(env.action_space, gym.spaces.box.Box):
            from baselines.TRPPO.KL2Clip_reduce_v3.KL2Clip_reduce import KL2Clip
            kl2clip = KL2Clip(dim=ac_space.shape[0], batch_size=nsteps, use_tabular=args.use_tabular)
        elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
            from baselines.TRPPO.KL2Clip_discrete.KL2Clip_discrete import KL2Clip
            kl2clip = KL2Clip(dim=ac_space.n, batch_size=nsteps, use_tabular=args.use_tabular)
        else:
            raise NotImplementedError('Please run atari or mujoco!')
        assert not (args.cliprange is None and args.delta_kl is None)
        if args.cliprange is None and args.delta_kl is not None:
            args.cliprange = kl2clip.get_cliprange_by_delta(args.delta_kl)
            print('********************************')
            print(f'We set cliprange={args.cliprange} according to delta_kl={args.delta_kl}, dim={ac_space.shape[0]}')

    cliprange = args.cliprange
    if isinstance(cliprange, float) or cliprange is None:
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)

    nupdates = total_timesteps // nbatch
    eprewmean_max = -np.inf
    alphas_kl2clip_decay = np.zeros(nupdates, dtype=np.float32)
    alphas_kl2clip_decay[0:nupdates // 3] = 1
    alphas_kl2clip_decay[nupdates // 3:] = np.linspace(1, -0.5, nupdates - nupdates // 3)

    for update in range(1, nupdates + 1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        if isinstance(env.action_space, gym.spaces.Box):
            cliprangenow = cliprange(frac)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            cliprangenow = (lambda _: cliprange(None) * _)(frac)  # anneal

        # using runner to sample data from model (old Pi_theta)
        obs, returns, masks, actions, values, neglogpacs, policyflats, states, epinfos = runner.run()  # pylint: disable=E0632
        # Add xiaoming
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        if isinstance(env.action_space, gym.spaces.box.Box):
            epinfobuf.clear()
        epinfobuf.extend(epinfos)
        mblossvals = []

        if clipped_type == ClipType.kl2clip:
            if isinstance(env.action_space, gym.spaces.Discrete):
                pas = np.exp(-neglogpacs)
            else:
                pas = None
            ress = kl2clip(
                mu0_logsigma0_cat=policyflats, a=actions, pas=pas,
                delta=args.delta_kl, clipcontroltype=args.kl2clip_clipcontroltype, cliprange=cliprange, )
            cliprange_max = ress.ratio.max
            cliprange_min = ress.ratio.min

            save_vars(osp.join(args.model_dir, 'cliprange_max', f'{update}'), cliprange_max)
            save_vars(osp.join(args.model_dir, 'cliprange_min', f'{update}'), cliprange_min)

            if not args.model_dir.__contains__('Humanoid') and isinstance(env.action_space, gym.spaces.box.Box):
                save_vars(osp.join(args.model_dir, 'actions', f'{update}'), actions)
        elif clipped_type == ClipType.judgekl:
            pass
        if not args.model_dir.__contains__('Humanoid') and isinstance(env.action_space, gym.spaces.box.Box):
            save_vars(osp.join(args.model_dir, 'mu0_logsigma0', f'{update}'), policyflats)
        save_vars(osp.join(args.model_dir, 'advs', f'{update}'), advs)

        if states is None:  # nonrecurrent version
            inds = np.arange(nbatch)
            kls = []
            ratios = []
            for ind_epoch in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    # mini-batch indexes
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, advs))
                    policyflats_batch, = (arr[mbinds] for arr in (policyflats,))
                    if clipped_type == ClipType.kl2clip:
                        cliprange_min_batch, cliprange_max_batch = (arr[mbinds] for arr in
                                                                    (cliprange_min, cliprange_max))
                        if isinstance(env.action_space, gym.spaces.Discrete):
                            cliprange_min_batch = 1 - (1. - cliprange_min_batch) * frac  # anneal
                            cliprange_max_batch = 1 + (cliprange_max_batch - 1) * frac
                        *ress, kl, ratio = model.train(lrnow, *slices, cliprange=cliprangenow,
                                                       cliprange_min=cliprange_min_batch,
                                                       cliprange_max=cliprange_max_batch, policyflats=policyflats_batch)
                    else:
                        *ress, kl, ratio = model.train(lrnow, *slices, cliprange=cliprangenow,
                                                       policyflats=policyflats_batch)
                    if ind_epoch == noptepochs - 1:
                        kls.append(kl)
                        ratios.append(ratio)
                    mblossvals.append(ress)
            inds2position = {}
            for position, ind in enumerate(inds):
                inds2position[ind] = position
            inds_reverse = [inds2position[ind] for ind in range(len(inds))]
            kls, ratios = (np.concatenate(arr, axis=0)[inds_reverse] for arr in (kls, ratios))
            save_vars(osp.join(args.model_dir, 'kls, ratios', f'{update}'), kls, ratios)

        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    # TODO: KL2CLip
                    raise NotImplementedError
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        eprewmean_newmax = False
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update * nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
            logger.logkv('eprewmean', eprewmean)
            if eprewmean > eprewmean_max:
                eprewmean_newmax = True
                eprewmean_max = eprewmean
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)

            # using tensorboard to log these data in a loop
            # print(logger.Logger.CURRENT.name2val)
            summary = tf.Summary()
            for k, v in logger.Logger.CURRENT.name2val.items():
                summary.value.add(tag=k, simple_value=v)
            # [summary.value.add(tag=k, simple_value=v) for k, v in logger.Logger.DEFAULT.name2val.items()]
            timesteps = update * nbatch
            writer.add_summary(summary, global_step=timesteps)

            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1 or update == nupdates
                # or eprewmean_newmax  # TODO: atari
        ) and args.model_dir:
            checkdir = osp.join(args.model_dir, 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()
    return model


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

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
from toolsm import tools

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
import baselines.common.tf_util as U

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, cliptype, args):
        sess = tf.get_default_session()

        # if args.policy_type == 'MlpPolicyMy':
        #     additional_keys = ['hidden_sizes','num_sharing_layers','ac_fn', 'seperate_logstd']
        # elif args.policy_type == 'MlpPolicy':
        #     additional_keys = ['seperate_logstd']
        # else:
        #     additional_keys = []
        #
        # additional_args = {}
        # for k in additional_keys:
        #     additional_args[k] = getattr( args, k)

        additional_args = {}
        if args.policy_type == 'MlpPolicyExt':
            additional_args = dict(args=args)

        # There are two models just to set different batch size

        act_model = policy(sess, ob_space, ac_space, nbatch_act, nsteps=1, reuse=False, **additional_args)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps=nsteps, reuse=True, **additional_args)
        if args.n_eval_epsiodes > 0:
            self.eval_policy = policy(sess, ob_space, ac_space, args.n_eval_epsiodes, nsteps=1, reuse=True, **additional_args)
            # ------- For copy Policy
            # self.eval_policy = eval_policy = policy(sess, ob_space, ac_space, args.n_eval_epsiodes, nsteps=1, reuse=False, name='policy_eval', **additional_args)
            # params_eval = eval_policy.variables_trainable
            # placeholders_params= [ tf.placeholder( dtype=v.dtype, shape=v.shape ) for v in params_eval ]
            # self.assign_eval_policy = U.function(inputs=placeholders_params,outputs=[], updates=[tf.assign(v, v_input) for (v, v_input) in zip(params_eval, placeholders_params)])
            # self.get_params_train = U.function( inputs=[], outputs=act_model.variables_trainable )
            # self.get_params_eval = U.function(inputs=[], outputs=eval_policy.variables_trainable )
            # ------ End------


            # for p in params:
            #     print( p.name )
            # for p in eval_policy.variables_all:
            #     print(p.name)
            # exit()
            # tools.save_vars( '/media/d/e/baselines_workingon/baselines/ppo2_AdaClip/t/a.pkl', params )
            # exit()





        self.OBS = OBS = train_model.X
        self.ACTIONS = ACTIONS = train_model.pdtype.sample_placeholder([None])

        self.RETURNS = RETURNS = tf.placeholder(tf.float32, [None])
        self.NEGLOGPACS_OLD = NEGLOGPACS_OLD = tf.placeholder(tf.float32, [None])
        self.VALUES_OLD = VALUES_OLD = tf.placeholder(tf.float32, [None])
        self.LR =  LR = tf.placeholder(tf.float32, [])
        self.CLIPRANGE =  CLIPRANGE = tf.placeholder(tf.float32, [])
        self.KLRANGE = KLRANGE = tf.placeholder(tf.float32, [])
        self.CLIPRANGE_LOWER = CLIPRANGE_LOWER = tf.placeholder(tf.float32, [None])
        self.CLIPRANGE_UPPER = CLIPRANGE_UPPER = tf.placeholder(tf.float32, [None])
        self.KL_COEF = KL_COEF = tf.placeholder(tf.float32, [])
        self.RANGE = RANGE = tf.placeholder(tf.float32, [])


        neglogpac = train_model.pd.neglogp(ACTIONS)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = VALUES_OLD + tf.clip_by_value(train_model.vf - VALUES_OLD, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - RETURNS)
        vf_losses2 = tf.square(vpredclipped - RETURNS)
        vf_loss = tf.maximum(vf_losses1, vf_losses2)
        vf_loss = .5 * tf.reduce_mean(vf_loss)

        ratio = tf.exp(NEGLOGPACS_OLD - neglogpac)

        pg_loss = None

        self.ADVS = ADVS = tf.placeholder(tf.float32, [None])

        new_pd = train_model.pd
        flat_shape = new_pd.flatparam().shape
        self.POLICYFLATS_OLD = POLICYFLATS_OLD = tf.placeholder(tf.float32, shape=flat_shape, name='old_policyflat')
        old_pd = train_model.pdtype.pdfromflat(POLICYFLATS_OLD)
        kl = old_pd.kl(new_pd)
        if hasattr(old_pd, 'wasserstein'):
            wasserstein = old_pd.wasserstein(new_pd)
        if hasattr(old_pd, 'totalvariation'):
            totalvariation = old_pd.totalvariation(new_pd)
        if cliptype in [ClipType.kl2clip, ClipType.adaptiverange_advantage]:
            pg_losses = -ADVS * ratio
            pg_losses2 = -ADVS * tf.clip_by_value(ratio, CLIPRANGE_LOWER, CLIPRANGE_UPPER)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            clipfrac = tf.reduce_mean(
                tf.to_float(tf.where(ADVS >= 0, tf.greater(ratio, CLIPRANGE_UPPER), tf.less(ratio, CLIPRANGE_LOWER))))
        else:
            raise NotImplementedError

        # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        approxkl = tf.reduce_mean(kl)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train( **kwargs):

            feed_dict = dict()
            for key, value in kwargs.items():
                feed_dict.update({ getattr(self,key.upper()): value})
            # td_map = {
            #     CLIPRANGE: cliprange,
            #     OBS: obs, ACTIONS: actions, ADV: advs, RETURNS: returns, LR: lr,
            #     NEGLOGPACS_OLD: neglogpacs, VALUES_OLD: values,
            #     POLICYFALTS_OLD: policyflats
            # }
            #
            # if cliptype == ClipType.kl2clip:
            #     assert cliprange_lower is not None and cliprange_upper is not None
            #     td_map.update({CLIPRANGE_LOWER: cliprange_lower, CLIPRANGE_UPPER: cliprange_upper})
            # elif cliptype == ClipType.adaptivekl:
            #     assert kl_coef is not None
            #     td_map.update({KL_COEF: kl_coef})


            # TODO: train_model.S .modify to STATES
            # recurrent version
            # if states is not None:
            #     td_map[train_model.S] = states
            #     td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, kl, ratio, _train],
                feed_dict
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']


        # def restore_summary_writer(graph_dir: str) -> tf.summary.FileWriter:
        #     return tf.summary.FileWriter.reopen()

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
        tf.global_variables_initializer().run(session=sess)  # pylint: disable=E1101





class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_policyflats = [], [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        # mb_obs_next = []
        mb_obs_next= None
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs, policyflats = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_policyflats.append(policyflats)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            # mb_obs_next.append( self.env._obs_real.copy() )

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        # mb_obs_next = np.asarray(mb_obs_next, dtype=np.float32)
        # --- End xiaoming
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_policyflats = np.asarray(mb_policyflats, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1: # Last Timestep
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



    # gradient_rectify = 7

from baselines.ppo2_AdaClip.algs import *
from toolsm import tools
def learn(*, policy, env, env_eval, n_steps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=1, nminibatches=4, n_opt_epochs=4,
          save_interval=10, load_path=None, cliptype=None, args=None):




    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * n_steps
    nbatch_train = nbatch // nminibatches

    set_save_load_env(env)
    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=n_steps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm, cliptype=cliptype, args=args)
    if save_interval:
        # only save make_model, the function to make a model with its closure/args
        import cloudpickle
        with open(osp.join(args.log_dir, 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    if load_path is not None:
        model.load(load_path)

    runner = Runner(env=env, model=model, nsteps=n_steps, gamma=gamma, lam=lam)

    if args.n_eval_epsiodes > 0:
        evaluator = Evaluator()

    # tfwriter = tf.summary.FileWriter(args.log_dir)

    if cliptype in [ClipType.kl2clip, ClipType.kl2clip_rollback]:
        # from baselines.ppo2_AdaClip.KL2Clip.KL2Clip_opt_tf import get_clip_new

        if isinstance(env.action_space, gym.spaces.box.Box):
            from baselines.ppo2_AdaClip.KL2Clip_reduce_v3.KL2Clip_reduce import KL2Clip, Adjust_Type
            kl2clip = KL2Clip(opt1Dkind=args.clipargs.kl2clip_opttype)

            args.clipargs.adjusttype = Adjust_Type[args.clipargs.adjusttype]
            if '2constant' in args.clipargs.adaptive_range:  # TODO: alg args
                dim = ac_space.shape[0]
                if args.clipargs.adjusttype == Adjust_Type.origin:
                    delta = args.clipargs.delta
                else:
                    delta = kl2clip.cliprange2delta(args.clipargs.cliprange, ac_space.shape[0],
                                                    args.clipargs.adjusttype)

                cliprange_upper_min = 1 + kl2clip.delta2cliprange(delta, dim=ac_space.shape[0],
                                                                  adjusttype='base_clip_upper')
                cliprange_lower_max = 1 - kl2clip.delta2cliprange(delta, dim=ac_space.shape[0],
                                                                  adjusttype='base_clip_lower')
            if args.clipargs.cliprange is None:
                assert isinstance(env.action_space, gym.spaces.box.Box)
                args.clipargs.cliprange = kl2clip.delta2cliprange(args.clipargs.klrange, dim=ac_space.shape[0],
                                                                  adjusttype=args.clipargs.adjusttype)
                tools.print_(
                    f'The provided cliprange is None. Set cliprange={args.clipargs.cliprange} by klrange={args.clipargs.klrange}, dim={ac_space.shape[0]}',
                    color='magenta')
        elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
            # TODO: Atari上cliprange是不断减小的,那么delta也应该不断减小
            # raise NotImplementedError('Please review the code.....')
            from baselines.ppo2_AdaClip.KL2Clip_discrete.KL2Clip_discrete import KL2Clip
            kl2clip = KL2Clip(opt1Dkind=args.clipargs.kl2clip_opttype)
        else:
            raise NotImplementedError('Please run atari or mujoco!')
    elif cliptype == ClipType.adaptivekl:
        kl_coef = 1.
        kl_targ = args.clipargs.klrange



    if args.envtype == MUJOCO:
        cliprange = args.clipargs.cliprange
    elif args.envtype == ATARI:
        cliprange = lambda f: f*args.clipargs.cliprange if args.clipargs.cliprange is not None else None

    if isinstance(cliprange, float) or cliprange is None:
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)


    # alphas_kl2clip_decay = np.zeros(nupdates, dtype=np.float32)
    # alphas_kl2clip_decay[0:nupdates // 3] = 1
    # alphas_kl2clip_decay[nupdates // 3:] = np.linspace(1, -0.5, nupdates - nupdates // 3)
    from toolsm.logger import Logger
    logformats = ['csv','tensorflow', 'log']
    if not args.is_multiprocess:
        logformats.append('stdout')
    else:
        logger_multiprocess = Logger( ['stdout'])
    logger = Logger( logformats , path=args.log_dir, file_basename='process' )

    epinfobuf = deque(maxlen=100)

    nupdates = total_timesteps // nbatch
    performance_max = -np.inf
    print(f'nupdates:{nupdates},eval_interval:{args.eval_interval}')
    tstart_ini = time.time()
    for update in range(1, nupdates + 1):
        tstart = time.time()
        assert nbatch % nminibatches == 0
        debugs = dict( iteration=update )
        nbatch_train = nbatch // nminibatches

        frac = (update-1) * 1. / nupdates
        # frac_remain = 1.0 - (update - 1.0) / nupdates
        frac_remain = 1.0 - frac
        lrnow = lr(frac_remain)

        # ---------------- Sample data
        if args.lam_decay:
            runner.lam = lam - (lam - 0.5) * frac
            print(f'lam decay to {runner.lam}')
        # ----- explore by setting policy std
        # if nbatch * update <= args.explore_timesteps:
        #     model.train_model.set_logstd(model.train_model.get_logstd() * 0)
        # if args.explore_additive_threshold is not None and update * 1. / nupdates > args.explore_additive_threshold:
        #     logger.log(f'add additive explore: {args.explore_additive_rate}')
        #     model.train_model.set_logstd(model.train_model.get_logstd() - args.explore_additive_rate)

        # with tools.timed('sample'):
        obs, returns, masks, actions, values, neglogpacs, policyflats, states, epinfos = runner.run()  # pylint: disable=E0632
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        debugs['advs'] = advs
        if isinstance(env.action_space, gym.spaces.Box):
            epinfobuf.clear()
        epinfobuf.extend(epinfos)


        # ----------------- Prepare for training: update clipping range, etc.
        cliprangenow = cliprange(frac_remain)
        # old version: unknown meaning....TODO: figure out the meaning
        # if isinstance(env.action_space, gym.spaces.Box):
        #     cliprangenow = cliprange(frac)
        # elif isinstance(env.action_space, gym.spaces.Discrete):
        #     cliprangenow = (lambda _: cliprange(None) * _)(frac)
        kwargs_in_scalar = dict(lr=lrnow, cliprange=cliprangenow)
        kwargs_in_arr = dict(
            obs=obs, returns=returns,
            actions=actions, values_old=values, neglogpacs_old=neglogpacs, advs=advs,
            policyflats_old=policyflats
        )
        if cliptype in [ClipType.kl2clip, ClipType.kl2clip_rollback]:
            pas = np.exp(-neglogpacs)

            if isinstance(env.action_space, gym.spaces.box.Box):
                results_kl2clip = kl2clip(
                    mu0_logstd0=policyflats, a=actions, pas=pas,
                    delta=args.clipargs.klrange,
                    adjusttype=args.clipargs.adjusttype, cliprange=args.clipargs.cliprange,
                    require_sol = False,
                    verbose = 1
                    # sharelogstd=args.clipargs, clip_clipratio=args.kl2clip_clip_clipratio,
                )
                cliprange_upper = results_kl2clip.ratio.max
                cliprange_lower = results_kl2clip.ratio.min

                if args.clipargs.adaptive_range == '2constant':
                    cliprange_upper = cliprange_upper - (cliprange_upper - cliprange_upper_min) * frac
                    cliprange_lower = cliprange_lower + (cliprange_lower_max - cliprange_lower) * frac
                    # TODO: debug tmp
                    debugs['cliprange_upper_min'] = cliprange_upper_min
                    debugs['cliprange_lower_max'] = cliprange_lower_max
                elif args.clipargs.adaptive_range == '2cliprange_final':
                    cliprange_final = args.clipargs.cliprange_final
                    cliprange_upper = cliprange_upper - (cliprange_upper - (1 + cliprange_final)) * frac
                    cliprange_lower = cliprange_lower + ((1 - cliprange_final) - cliprange_lower) * frac
                elif args.clipargs.adaptive_range == '2constant_upper':
                    cliprange_upper = cliprange_upper - (cliprange_upper - cliprange_upper_min) * frac
                    # TODO: debug tmp
                    debugs['cliprange_upper_min'] = cliprange_upper_min
                    debugs['cliprange_lower_max'] = cliprange_lower_max
                # elif args.clipargs.adaptive_range == '2cliprange':
                #     # TODO: cliprange may be none....
                #     cliprange_upper = cliprange_upper - (cliprange_upper- (1+args.clipargs.cliprange))*frac
                #     cliprange_lower = cliprange_lower + ((1-args.clipargs.cliprange) - cliprange_lower )*frac
                elif args.clipargs.adaptive_range == 'clip2cliprange':
                    frac_threshold = args.clipargs.frac_threshold
                    if frac >= frac_threshold:
                        cliprange_upper[:] = 1 + args.clipargs.cliprange
                        cliprange_lower[:] = 1 - args.clipargs.cliprange
            elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
                results_kl2clip = kl2clip(
                    pas=pas,
                    delta=args.clipargs.klrange,
                    verbose = 1
                )
                cliprange_upper = results_kl2clip.ratio.max
                cliprange_lower = results_kl2clip.ratio.min

                cliprange_upper = 1 + (cliprange_upper - 1) * frac_remain
                cliprange_lower = 1 - (1. - cliprange_lower) * frac_remain

            # if isinstance(env.action_space, gym.spaces.Discrete):
            #     raise NotImplemented('Please review the code')
            #     cliprange_upper = 1 + (cliprange_upper - 1) * frac_remain
            #     cliprange_lower = 1 - (1. - cliprange_lower) * frac_remain


            debugs['cliprange_upper'] = cliprange_upper
            debugs['cliprange_lower'] = cliprange_lower
            kwargs_in_arr.update(
                cliprange_upper=cliprange_upper,
                cliprange_lower=cliprange_lower,
            )



        # print(kwargs_in_scalar)
        # ----------------- Train the model
        mblossvals = []
        if states is None:  # nonrecurrent version
            kls = []
            ratios = []
            # totalvariations = []

            inds = np.arange(nbatch)

            for ind_epoch in range(n_opt_epochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    mbinds = inds[start: start + nbatch_train]
                    kwargs_in_batch = dict()
                    for key in kwargs_in_arr.keys():
                        kwargs_in_batch[key] = kwargs_in_arr[key][mbinds]

                    *lossvals, kl, ratio = model.train(**kwargs_in_scalar,**kwargs_in_batch)
                    mblossvals.append(lossvals)
                    if ind_epoch == n_opt_epochs -1:# only add it at the last opt_epoch
                        kls.append(kl)
                        ratios.append(ratio)
                        # totalvariations.append(totalvariation)
            # --- restore the order of kls and ratios to the original order
            # kls and ratios are list of kl_batch
            # TODO: Do not suffle and run it at last time! make it easy to add more variables to debug!
            inds2position = {}
            for position, ind in enumerate(inds):
                inds2position[ind] = position
            inds_reverse = [inds2position[ind] for ind in range(len(inds))]
            kls, ratios = (np.concatenate(arr, axis=0)[inds_reverse] for arr in (kls, ratios))
            debugs['kls'] = kls
            debugs['ratios'] = ratios
            # debugs['totalvariations'] = totalvariations
            # print(kls.mean(), totalvariations.mean())


        else:  # recurrent version
            raise NotImplementedError('Not implemented!')
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * n_steps).reshape(nenvs, n_steps)
            envsperbatch = nbatch_train // n_steps
            # TODO: states: masks
            for _ in range(n_opt_epochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
        if args.save_debug:
            tools.save_vars( osp.join(args.log_dir, 'debugs.pkl'), debugs, append=update > 1 )

        # -------------- Log the result

        lossvals = np.mean(mblossvals, axis=0)
        is_eprewmean_better = False
        eprewmean_eval = None
        eplenmean_eval = None
        if args.n_eval_epsiodes > 0 and \
            (
                (args.eval_interval >0 and ( (update - 1) % args.eval_interval == 0) )
                    or
                (args.eval_interval<0 and update == nupdates)
            ):
            result_eval = evaluator.eval( env_eval, env, model, args, update )
            eprewmean_eval = safemean([epinfo['r'] for epinfo in result_eval])
            eplenmean_eval = safemean([epinfo['l'] for epinfo in result_eval])
            if eprewmean_eval > performance_max:
                is_eprewmean_better = True
                performance_max = eprewmean_eval

        if save_interval and ( (update -1 ) % save_interval == 0 or update == nupdates or is_eprewmean_better
        ):
            savepath = osp.join(args.model_dir, '%.5i' % update)
            # print('Saving to', savepath)
            model.save(savepath)

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
            eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            timesteps = update * nbatch
            tnow = time.time()
            time_oneupdate = tnow - tstart
            fps = int(nbatch / time_oneupdate)

            if args.is_multiprocess: #and ( (update-1) % (args.eval_interval*2) == 0)
                # TODO: 改成log, print reward_eval
                logger_multiprocess.log_keyvalues(
                    update=update,
                    env=args.env,
                    timesteps=timesteps,
                    time_oneupdate=time_oneupdate,
                    eprewmean=eprewmean,
                    eprewmean_eval=eprewmean_eval,
                    time = tools.time_now_str('%H:%M:%S')
                )
                # tools.print_( f'timesteps:{timesteps},time_oneupdate:{time_oneupdate},eprewmean:{eprewmean},eprewmean_eval:{eprewmean_eval}', color='magenta' )

            logger.log_keyvalue(
                global_step = timesteps,
                nupdates = update,
                fps = fps,
                eprewmean = eprewmean,
                eplenmean = eplenmean,
                eprewmean_eval = eprewmean_eval,
                eplenmean_eval=eplenmean_eval,
                explained_variance = float(ev),
                time_elapsed = tnow - tstart_ini,
                time_oneupdate = time_oneupdate,
                serial_timesteps =update * n_steps,
                total_timesteps = timesteps,
            )
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.log_keyvalue( **{lossname:lossval} )
            
            logger.dump_keyvalues()


            # if args.debug_halfcheetah and args.env_pure == 'HalfCheetah':
            #     if frac > 0.6 and performance_max <= 1500:
            #         print( tools.colorize('HalfCheetah does not ahieve the threshold! Force it to stop!') )
            #         break

        if cliptype == ClipType.adaptivekl:
            kl_mean = np.mean(kls)
            if kl_mean < kl_targ / 1.5:
                kl_coef /= 2
            elif kl_mean > kl_targ * 1.5:
                kl_coef *= 2



    if args.model_dir is not None:
        import shutil
        for d in args.zip_dirs:
            d_path = args[f'{d}_dir']
            if len(tools.get_files( d_path ))> 2:
                shutil.make_archive( base_name=d_path, format='zip', root_dir=d_path )
                tools.safe_delete( d_path , confirm=False, require_not_containsub=False )

    env.close()
    return model


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



class Evaluator():
    def __init__(self):
        pass
        # # ---- Clone env
        # self.env_eval = env_eval
        # # ----- copy pi
        # self.model = model
        # self.assign_policy = model.assign_policy_eval_eq_train
        # self.policy = model.eval_policy


    def eval(self, env_eval, env_train, model, args, update):
        # from copy import deepcopy
        from copy import copy
        # assign env
        if args.envtype == MUJOCO:
            env_eval.ob_rms = copy(env_train.ob_rms)
        else:
            pass

        # assign pi
        # params = model.get_params_train()
        # model.assign_eval_policy( *params )

        eval_policy = model.eval_policy
        # start process
        epinfos = rollout( env_eval, eval_policy, evaluate_times=args.n_eval_epsiodes, deterministic=True, verbose=(args.envtype == ATARI) )
        # print(f'update:{update}, epinfos:{epinfos}')
        # rewards_epi = [ epinfo['r'] for epinfo in epinfos ]
        return epinfos


def rollout(env, policy, evaluate_times=1, deterministic=True, verbose=0):
    import itertools

    epinfos = []
    obs = env.reset()
    if verbose:
        tools.warn_(f'Evaluate Policy...')
    # rewards_episode = []
    for t in itertools.count():
        # tools.print_refresh(t)
        if not deterministic:
            actions, *_ = policy.step(obs)
        else:
            actions, *_ = policy.step_test(obs)

        obs[:], reward, dones, infos = env.step(actions)
        # env.render()
        # cnt_dones += dones.sum()
        # If it is done, it will contains a key 'episode' in info
        # {'episode': {'r': 118.048395, 'l': 64, 't': 0.67222}}
        # print(infos, f't={t},done={dones}')
        # if dones[0]:
        #     print('done')
        #     exit()
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)
        if len(epinfos) >= evaluate_times:
            break
    return epinfos




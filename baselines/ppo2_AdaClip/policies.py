import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, name='policy', **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        with tf.variable_scope(name, reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def step_test(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([tf.argmax(self.pd.logits, axis=-1), vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def step_policyflat(ob, *_args, **_kwargs):
            a, v, neglogp, polciyflat = sess.run([a0, vf, neglogp0, self.pd.logits], {X:ob})
            # a, v, self.initial_state, neglogp = self.step(ob, *_args, **_kwargs)
            # pa = np.exp(-neglogp)
            return a, v, self.initial_state, neglogp, polciyflat

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.step_test = step_test
        self.step_policyflat = step_policyflat
        self.value = value

class LnLstmMlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            # h = nature_cnn(processed_x)
            activ = tf.tanh
            h = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))

            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class LstmMlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            # h = nature_cnn(X)
            activ = tf.tanh
            h = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))

            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value


class LstmMlp20ChasePolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=64, reuse=False):
        nenv = nbatch // nsteps
        print(f'{nlstm}')
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            # h1 = fc(X, 'fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            xs = batch_to_seq(h1, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h2, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h2 = seq_to_batch(h2)
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        v0 = vf[:, 0]
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        def get_act(ob, state, mask):
            a = sess.run(a0, {X:ob, S:state, M:mask})
            return a

        def get_mean(ob, state, mask):
            a, state_new = sess.run([pi, snew], {X:ob, S:state, M:mask})
            return a, state_new


        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.act = get_act
        self.mean = get_mean


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, reuse=False, **kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def step_policyflat(ob, *_args, **_kwargs):
            a, v, neglogp, polciyflat = sess.run([a0, vf, neglogp0, self.pd.flat], {X:ob})
            return a, v, self.initial_state, neglogp, polciyflat

        def step_test(ob, *_args, **_kwargs):
            a = sess.run([self.pd.mean], {X:ob})
            return a

        self.X = X
        self.vf = vf
        self.step = step
        self.step_policyflat = step_policyflat
        self.value = value
        self.step_test = step_test


import baselines.common.tf_util as U
from gym.spaces import Discrete, Box

from enum import Enum

class ActionType( ):
    deterministic = 0
    stochastic = 1
    placeholder = 2


class MlpPolicyExt(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False,
                 name='policy', args=None): #pylint: disable=W0613
        policy_variance_state_dependent = args.policy_variance_state_dependent
        ac_fn = args.ac_fn
        hidden_sizes = args.hidden_sizes
        num_sharing_layers = args.num_sharing_layers
        num_layers = args.num_layers
        assert ac_fn in ['tanh', 'sigmoid', 'relu']

        if isinstance(hidden_sizes, int):
            assert num_layers is not None
            hidden_sizes = [hidden_sizes] * num_layers
        if num_layers is None:
            num_layers = len(hidden_sizes)
        assert num_layers == len(hidden_sizes)


        # print(f'Policy hidden_sizes:{hidden_sizes}')

        self.pdtype = make_pdtype(ac_space)

        with tf.variable_scope(name, reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)

            activ = getattr( tf.nn, ac_fn )
            processed_x = tf.layers.flatten(processed_x)

            # --- share layers
            for ind_layer in range(num_sharing_layers):
                processed_x = activ( fc(processed_x, f'share_fc{ind_layer}', nh=hidden_sizes[ind_layer], init_scale=np.sqrt(2)) )

            # --- policy
            pi_h = processed_x
            for ind_layer in range( num_sharing_layers, num_layers ):
                pi_h = activ(fc(pi_h, f'pi_fc{ind_layer}', nh=hidden_sizes[ind_layer], init_scale=np.sqrt(2)))

            from gym import spaces
            params_addtional = {}
            if policy_variance_state_dependent and isinstance( ac_space, spaces.Box ):
                latent_logstd = processed_x
                for ind_layer in range(num_sharing_layers, num_layers):
                    latent_logstd = activ(fc(latent_logstd, f'logstd_fc{ind_layer}', nh=hidden_sizes[ind_layer], init_scale=np.sqrt(2)))
                params_addtional['latent_logstd'] = latent_logstd

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h, init_scale=0.01, logstd_initial=args.logstd, **params_addtional)


            # --- value function
            vf_h = processed_x
            for ind_layer in range( num_sharing_layers, num_layers ):
                vf_h = activ(fc(vf_h, f'vf_fc{ind_layer}', nh=hidden_sizes[ind_layer], init_scale=np.sqrt(2)))
            vf = fc(vf_h, 'vf', 1)[:,0]



            a_sample = self.pd.sample()
            neglogp_sample = self.pd.neglogp(a_sample)
            self.initial_state = None


            # --- predict function
            # use placeholder
            # use stochastic action
            # use deterministic action
            if args.coef_predict_task > 0:
                import tensorflow.contrib.distributions as dists
                assert isinstance( ac_space, Box ), 'Only Implement for Box action space'
                A_type = tf.placeholder_with_default('pl', dtype=tf.string)
                A_pl = self.pdtype.sample_placeholder([None])
                self.A = A_pl
                self.A_type = A_type

                A_input_1 = U.switch( tf.equal( A_type, 'det' ), self.pd.mode(), a_sample )
                A_input = U.switch( tf.equal( A_type, 'pl' ), A_pl,A_input_1)
                predict_h = tf.concat( (processed_x, A_input))
                for ind_layer in range(num_sharing_layers, num_layers):
                    predict_h = activ(fc(predict_h, f'predict_fc{ind_layer}', nh=hidden_sizes[ind_layer], init_scale=np.sqrt(2)))
                predict_mean = fc(predict_h, f'predict_fc{ind_layer}', nh=ob_space.shape[0], init_scale=np.sqrt(2))

                predict_cov_init_value = np.identity( shape=ob_space.shape )
                predict_cov = tf.get_variable( name='predict_cov', shape=predict_cov_init_value, initializer=tf.constant_initializer(predict_cov_init_value) )
                predict_dist = dists.MultivariateNormalTriL( predict_mean, predict_cov )
                self.predict_dist = predict_dist

            scope_model = tf.get_variable_scope().name
            self.variables_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_model)
            self.variables_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_model)


        #--- set logstd
        if not policy_variance_state_dependent:
            assert isinstance( ac_space, Box ), 'Only work for Continuous Action Space'
            logstd_pl, _ = observation_input( ac_space, batch_size=1, name='ac' )
            assign_logstd = tf.assign( self.pdtype.logstd, logstd_pl )
            set_logstd_entity = U.function([logstd_pl], assign_logstd)
            def set_logstd(logstd_new):
                # if isinstance( logstd_new, float  ):
                #     logstd_new = [[logstd_new] * ac_space.shape[0]]
                set_logstd_entity(logstd_new)
            self.set_logstd = set_logstd

        self.get_logstd = U.function([], self.pdtype.logstd)

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a_sample, vf, neglogp_sample], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        def step_policyflat(ob, *_args, **_kwargs):
            a, v, neglogp, polciyflat = sess.run([a_sample, vf, neglogp_sample, self.pd.flatparam()], {X:ob}) #TODO: TEST flat for discrete action space
            return a, v, self.initial_state, neglogp, polciyflat

        def step_test(ob, *_args, **_kwargs):
            a = sess.run([self.pd.mode()], {X:ob})
            return a

        self.X = X
        self.vf = vf
        self.step = step
        self.step_policyflat = step_policyflat
        self.value = value
        self.step_test = step_test







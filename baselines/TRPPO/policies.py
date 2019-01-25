import numpy as np
import tensorflow as tf
from baselines.common.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
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


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def step_test(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([tf.argmax(self.pd.logits, axis=-1), vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def step_policyflat(ob, *_args, **_kwargs):
            a, v, neglogp, polciyflat = sess.run([a0, vf, neglogp0, self.pd.logits], {X: ob})
            # a, v, self.initial_state, neglogp = self.step(ob, *_args, **_kwargs)
            # pa = np.exp(-neglogp)
            return a, v, self.initial_state, neglogp, polciyflat

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.step_test = step_test
        self.step_policyflat = step_policyflat
        self.value = value


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def step_policyflat(ob, *_args, **_kwargs):
            a, v, neglogp, polciyflat = sess.run([a0, vf, neglogp0, self.pd.flat], {X: ob})
            return a, v, self.initial_state, neglogp, polciyflat

        def step_test(ob, *_args, **_kwargs):
            a = sess.run([self.pd.mean], {X: ob})
            return a

        self.X = X
        self.vf = vf
        self.step = step
        self.step_policyflat = step_policyflat
        self.value = value
        self.step_test = step_test

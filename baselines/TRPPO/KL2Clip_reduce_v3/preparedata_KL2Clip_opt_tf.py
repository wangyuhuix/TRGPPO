from pathos import multiprocessing
from baselines.common.tools import load_vars
from dotmap import DotMap

import os
import time

import numpy as np
from baselines.common import tools
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from multiprocessing import Pool

from baselines.common.distributions import DiagGaussianPd

import tensorflow as tf
from tensorflow.contrib.constrained_optimization import ConstrainedMinimizationProblem, AdditiveExternalRegretOptimizer

from baselines.common import plt_tools

from baselines.common.tools import save_vars

# path_root = '/root/d/e/et/baselines'
path_root = '/home/hugo/Desktop/wxm'


class MinimizationProblem0(ConstrainedMinimizationProblem):
    """A `ConstrainedMinimizationProblem` with constant constraint violations.

    This minimization problem is intended for use in performing simple tests of
    the Lagrange multiplier (or equivalent) update in the optimizers. There is a
    one-element "dummy" model parameter, but it should be ignored.
    """

    def __init__(self, inputs, constraints):
        self._objective = inputs
        self._constraints = constraints

    @property
    def objective(self):
        """Returns the objective function."""
        return self._objective

    @property
    def constraints(self):
        """Returns the constant constraint violations."""
        return self._constraints


def get_com_batch(dim, sess, batch_size, share_size, sharelogsigma):
    # 这里的batch_size 不指定 用None会有bug
    with sess.as_default(), sess.graph.as_default():
        pls = DotMap()
        x0 = tf.placeholder(dtype=tf.float32, shape=(batch_size, dim * 2), name='x0')
        if not sharelogsigma:
            x_initial = tf.placeholder(dtype=tf.float32, shape=x0.shape, name='x0')
            x = tf.Variable(x_initial, name='x')

            pls.x_initial = x_initial
        else:
            mu_initial = tf.placeholder(dtype=tf.float32, shape=(batch_size, dim), name='x0')
            independent_size = batch_size // share_size
            logsigma_initial = tf.placeholder(dtype=tf.float32, shape=(independent_size, dim), name='x0')
            mu = tf.Variable(mu_initial, name='mu')
            logsigma = tf.Variable(logsigma_initial, name='logsigma')
            logsigma_all = tf.tile(logsigma, [1, share_size])
            logsigma_all = tf.reshape(logsigma_all, [-1, dim])
            x = tf.concat((mu, logsigma_all), axis=-1)

            pls.mu_initial = mu_initial
            pls.logsigma_initial = logsigma_initial

        a = tf.placeholder(dtype=tf.float32, shape=(batch_size, dim), name='a')
        delta = tf.placeholder(dtype=tf.float32, shape=(), name='delta')

        # --- objective function
        dist = DiagGaussianPd(x)
        f = dist.neglogp(a)
        p = dist.p(a)
        dist0 = DiagGaussianPd(x0)
        # con = dist.kl(dist0) - delta
        con = dist0.kl(dist) - delta  # 拟合 lambda

        p0 = dist0.p(a)
        ratio = p / p0

        pls_new = DotMap(
            x0=x0,
            a=a,
            delta=delta
        )
        pls.update(pls_new)
        ffs = DotMap(
            p=p,
            p0=p0
        )
        return f, con, ratio, x, ffs, pls


def get_ConstraintThreshold(delta):
    return delta * ConstraintThreshold_Multiplier


import warnings


class KL2Clip(object):
    def __init__(self, dim, batch_size, sharelogsigma, clipcontroltype, cliprange):
        if batch_size != 2048:
            warnings.warn(
                f'Current optimizer is optimized for batch_size=2048! \n If you still want to use bacth_size={batch_size}, the result of ratio_min may not be right!')
        sess = KL2Clip.create_session()

        # gradOpts = [tf.train.AdagradOptimizer, tf.train.AdamOptimizer]
        # gradOpts = [tf.train.AdagradOptimizer, tf.train.AdagradOptimizer]
        gradOpts = [tf.train.AdagradOptimizer]
        # gradOpts = [tf.train.GradientDescentOptimizer]
        num_gradOpts = len(gradOpts)
        num_objectives = 2
        assert 1 <= num_objectives and num_objectives <= 2
        # assert len(gradOpts) <=2

        self.sess = sess
        # --- obtain raw computed graphs
        fs, cons, ratios, xs, ffs, pls = get_com_batch(dim, sess, num_gradOpts * num_objectives * batch_size,
                                                       batch_size, sharelogsigma)
        ps = ffs.p

        # --- obtain the idxs that perform better
        def to2d(v_):
            shape = list(v_.shape)
            shape[0] = shape[0] // num_gradOpts
            return tf.reshape(v_, [num_gradOpts] + shape)

        fs_ori, cons, ratios, ps, xs = [to2d(v__) for v__ in (fs, cons, ratios, ps, xs)]

        # --- Trick: decay learning_rate
        global_step = tf.Variable(0, trainable=False, name='global_step')

        def get_learningrate():
            LearningRateMax = 0.05
            LearningRateMin = 0.01
            blocks = [500 + 100 * i for i in range(6)]
            boundaries = blocks[0:-1]
            for i in range(1, len(blocks) - 1):
                boundaries[i] = boundaries[i - 1] + blocks[i]
            values = np.linspace(LearningRateMax, LearningRateMin, len(blocks)).tolist()
            self.blocks = blocks
            if DEBUG:
                print(f'blocks:{blocks}, boundaries:{boundaries}, values:{values}')
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)
            return learning_rate

        learning_rate = get_learningrate()
        ConstraintThreshold = get_ConstraintThreshold(pls.delta)
        opt_ops = []
        fs = [None] * num_gradOpts
        for i, opt_grad in enumerate(gradOpts):
            # --- modify the minimize function
            f_max, f_min = tf.split(fs_ori[i], num_or_size_splits=num_objectives, axis=0)
            fs[i] = tf.concat((f_max, tf.negative(f_min)), axis=0)
            # --- generate opt_op
            problem = MinimizationProblem0(inputs=fs[i], constraints=cons[i])
            optimizer = AdditiveExternalRegretOptimizer(optimizer=opt_grad(learning_rate=learning_rate))
            if i == 0:
                op = optimizer.minimize(minimization_problem=problem, global_step=global_step)
            else:
                op = optimizer.minimize(minimization_problem=problem)
            opt_ops.append(op)

        if num_gradOpts == 1:
            assert ratios.shape[0] == 1
            f_all = fs[0]
            ratio_all = ratios[0]
            p_all = ps[0]
            con_all = cons[0]
            x_all = xs[0]
        elif num_gradOpts == 2:  # Trick: multiple gradOptimizers
            fs = tf.stack(fs, axis=0)
            # fs: 2×(batchsize*2)
            # fs[i]: the objective function f of i-th optimizer,
            #   shape:(batchsize*2)
            #       from 0 to (batchsize-1): max,
            #       from batchsize to batchsize*2-1: min
            # cons[i]: the constraint con of i-th optimizer,
            '''
                if con0<threshold and con1<threshold: 这种比较方法只能进行两个gradOpt的比较
                    compare(f0, f1)
                else:
                    compare(con0, con1)
            '''
            cons_condition = cons < ConstraintThreshold
            both_satisfy = tf.logical_and(cons_condition[0], cons_condition[1])
            both_satisfy = tf.tile(tf.reshape(both_satisfy, (1, -1)), (num_gradOpts, 1))
            measure = tf.where(both_satisfy, fs, cons)  # if both satisfy: compare f; else compare con.
            idx = tf.stack((tf.argmin(measure), tf.range(num_objectives * batch_size, dtype=tf.int64)), axis=1)
            # --- filter the output by idxs which have better output
            ratio_all = tf.gather_nd(ratios, idx)
            p_all = tf.gather_nd(ps, idx)
            con_all = tf.gather_nd(cons, idx)
            x_all = tf.gather_nd(xs, idx)
        else:
            raise NotImplementedError

        # --- Trick: decide when to stop
        # 注意仅仅只考虑 con_max,因为con_min有些解无论怎么优化都没有办法满足约束
        with tf.control_dependencies(opt_ops):

            def split2max_min(obj_):
                return tf.split(obj_, num_or_size_splits=num_objectives, axis=0)

            ratio_max, ratio_min = split2max_min(ratio_all)
            p_max, p_min = split2max_min(p_all)
            con_max, con_min = split2max_min(con_all)
            # x_all = tf.identity(x_all)
            x_max, x_min = split2max_min(x_all)

            # Trick: set ratio_min=1/ratio_max when it is zero
            # ratio_min = tf.where(ratio_min < 1e-10, 1. / ratio_max, ratio_min)
            # TODO: tmp: set ratio_min=1/ratio_max direcyly
            if not sharelogsigma:
                ratio_min = tf.where(ratio_min < 1e-10, 1. / ratio_max, ratio_min)
            else:
                ratio_min = 1. / ratio_max

            rate_satisfycon = tf.reduce_mean(tf.to_double(con_max <= ConstraintThreshold))
            rate_satisfyratio_max = tf.reduce_mean(tf.to_double(ratio_max > 1.))
            rate_satisfyratio_min = tf.reduce_mean(tf.to_double(ratio_min < 1.))

            # x_max_old = tf.placeholder( dtype=tf.float32, shape=x_max.shape, name='x_max_old' )
            # difference = tf.sqrt( tf.reduce_mean( tf.square( x_max - x_max_old ), axis=1 ) )
            # tf.reduce_all(difference <=)
            ratio_max_old = tf.placeholder(dtype=tf.float32, shape=ratio_max.shape)
            ratio_min_old = tf.placeholder(dtype=tf.float32, shape=ratio_min.shape)
            pls.ratio_max_old = ratio_max_old
            pls.ratio_min_old = ratio_min_old
            ratio_old = tf.concat((ratio_max_old, ratio_min_old), axis=0)
            # difference = tf.abs(ratio - ratio_old)/tf.maximum( tf.abs(ratio_old), 0.1 )
            difference = tf.abs(ratio_max - ratio_max_old) / tf.maximum(tf.abs(ratio_max_old),
                                                                        0.1)  # TODO: Deleted Ratio_min
            difference_max = tf.reduce_max(difference)
            rate_statisfydifference = tf.reduce_mean(tf.to_double(difference <= Difference_Threshold))
        # ratio_max_test  = DiagGaussianPd(x_max).p( pls.a[:batch_size] )/DiagGaussianPd(pls.x0[:batch_size]).p( pls.a[:batch_size] )
        # ratio_min_test = DiagGaussianPd(x_min).p(pls.a[:batch_size]) / DiagGaussianPd(pls.x0[:batch_size]).p(pls.a[:batch_size])

        intial_variables = tf.global_variables_initializer()

        def opt(mu0_logsigma0_cat=None, mu0_logsigma0_tuple=None, delta=None, a=None, clip_clipratio=None):
            if delta is None:
                delta = Setting[sharelogsigma][clipcontroltype][cliprange].dim2delta[dim]
                print(
                    f'delta is None, choose delta={delta} with default setting of sharelogsigma={sharelogsigma}, clipcontroltype={clipcontroltype}, cliprange={cliprange}')
            if mu0_logsigma0_tuple is not None:
                mu0, logsigma0 = mu0_logsigma0_tuple
                mu0_logsigma0_cat = np.concatenate((mu0, logsigma0), axis=-1)
            elif mu0_logsigma0_cat is not None:
                mu0 = mu0_logsigma0_cat[:, :dim]
                logsigma0 = mu0_logsigma0_cat[:, -dim:]
            else:
                raise Exception('Error')
            if a is None:
                a = np.zeros((mu0_logsigma0_cat.shape[0], dim), dtype=np.float32)

            assert delta is not None
            assert mu0_logsigma0_cat.shape[1] // 2 == dim
            assert (logsigma0 == logsigma0[
                0]).all(), 'Now we assume all the logsigma are equal, this condition are used some where'

            if indBlock is not None:
                NumSteps = [np.sum(self.blocks[0:indBlock]), np.sum(self.blocks), 8000]
            else:
                NumSteps = [0, np.sum(self.blocks), 8000]
            for i in range(1, len(NumSteps)):
                assert NumSteps[i] > NumSteps[i - 1]
            if DEBUG:
                print('blocks', self.blocks, 'NumSteps', NumSteps)
            '''
                NumSteps have three level
                0-th: force to optimize
                1-th: continue to optimize if some conditions does not satisfy
                2-th: continue to optimize if some conditions does not satisfy (less strict)
            '''
            with sess.as_default(), sess.graph.as_default():
                if not sharelogsigma:
                    feed_dict = {
                        pls.x0: np.tile(mu0_logsigma0_cat, [num_gradOpts * num_objectives, 1]),
                        pls.a: np.tile(a, [num_gradOpts * num_objectives, 1]),
                        pls.delta: delta,
                        pls.x_initial: np.tile(mu0_logsigma0_cat, [num_gradOpts * num_objectives, 1]),
                    }
                else:
                    feed_dict = {
                        pls.x0: np.tile(mu0_logsigma0_cat, [num_gradOpts * num_objectives, 1]),
                        pls.a: np.tile(a, [num_gradOpts * num_objectives, 1]),
                        pls.delta: delta,
                        pls.mu_initial: np.tile(mu0, [num_gradOpts * num_objectives, 1]),
                        pls.logsigma_initial: np.tile(logsigma0[0], [num_gradOpts * num_objectives, 1])
                    }
                sess.run(intial_variables, feed_dict)
                for step in range(NumSteps[0]):
                    sess.run(opt_ops, feed_dict)
                ratio_max_, ratio_min_ = sess.run([ratio_max, ratio_min], feed_dict=feed_dict)
                for step in range(NumSteps[0], NumSteps[2]):
                    step += 1
                    feed_dict[pls.ratio_max_old] = ratio_max_
                    feed_dict[pls.ratio_min_old] = ratio_min_
                    *_, rate_satisfycon_, rate_satisfyratio_max_, rate_satisfyratio_min_, \
                    ratio_max_, ratio_min_, p_max_, p_min_, con_max_, con_min_, rate_statisfydifference_, \
                    x_max_, x_min_, \
                    difference_max_, \
                        = \
                        sess.run(opt_ops +
                                 [rate_satisfycon, rate_satisfyratio_max, rate_satisfyratio_min, ratio_max, ratio_min,
                                  p_max, p_min, con_max, con_min, rate_statisfydifference, x_max, x_min,
                                  difference_max],
                                 feed_dict)
                    if \
                            (rate_satisfycon_ >= Rate_StatisfyConstraint or step >= NumSteps[1]) \
                                    and rate_satisfyratio_max_ >= Rate_StatisfyRatio \
                                    and rate_statisfydifference_ >= Rate_Statisfydifference \
                            :
                        # and rate_satisfyratio_min_ >= Rate_StatisfyRatio \ # TODO: tmp Deleted Ratio_Min
                        break
                # tools.print_time()
                # vs= sess.run([ fs, cons, cons_condition, idx  ], feed_dict)
                # save_vars( 'a.pkl', x0_batch, *vs )
                # ratio_max_r, ratio_min_r, con_max_r, con_min_r = \
                # sess.run([ratio_max, ratio_min, con_max, con_min], feed_dict)
                if DEBUG:
                    print(
                        f'step:{step}, rate_satisfyconstraint:{rate_satisfycon_}, rate_satisfyratio_max:{rate_satisfyratio_max_},rate_satisfyratio_min:{rate_satisfyratio_min_}, NumSteps:{NumSteps}, {(ratio_max_>=1).all()}')
                # save_vars('a.pkl',ratio_max_r, ratio_min_r, con_max_r, con_min_r , rate_satisfycon_, rate_satisfyratio_max_, rate_satisfyratio_min_)
                if DEBUG:
                    # TODO: 这里始终有个错误,就是这两个计算的值始终有点不一样
                    pass
                    # p_max_x = DiagGaussianPd(x_max_).p(a).eval()
                    # print( p_max_x[0:10],'\n', p_max_[0:10] )
                if clip_clipratio is not None:
                    assert clip_clipratio >= 1
                    clip_clipratio_new = ratio_max_.min() - 1 + clip_clipratio
                    print(f'clip clipratio to {clip_clipratio_new}')

                    ratio_max_ = np.minimum(ratio_max_, clip_clipratio_new)
                    ratio_min_ = np.maximum(ratio_min_, 1. / clip_clipratio_new)
                return DotMap(
                    ratio=DotMap(max=ratio_max_, min=ratio_min_),
                    p=DotMap(max=p_max_, min=p_min_),
                    con=DotMap(max=con_max_, min=con_min_),
                    x=DotMap(max=x_max_, min=x_min_),
                    # x_all = x_all_,
                    # p0 = p0_,
                    step=step,
                    rate_satisfycon_=rate_satisfycon_,
                    rate_satisfyratio_max_=rate_satisfyratio_max_,
                    rate_statisfydifference_=rate_statisfydifference_,
                    difference_max_=difference_max_,
                    rate_satisfyratio_min_=rate_satisfyratio_min_,
                    delta=delta
                )

        self._opt = opt

    def __call__(self, mu0_logsigma0_cat=None, mu0_logsigma0_tuple=None, delta=None, a=None, clip_clipratio=None):
        return self._opt(mu0_logsigma0_cat, mu0_logsigma0_tuple, delta, a, clip_clipratio)

    def close(self):
        self.sess.close()
        ops.reset_default_graph()

    @staticmethod
    def create_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess


import baselines.common.tf_util as U

from mpl_toolkits.mplot3d import axes3d


def arr2meshgrid(arr, dim):
    meshgrids = [arr for i in range(dim)]
    return multiarr2meshgrid(meshgrids)


def multiarr2meshgrid(meshgrids):
    arr = np.meshgrid(*meshgrids)
    arr = [i.reshape(-1, 1) for i in arr]
    arr = np.concatenate(arr, axis=1)
    return arr


def normalize_clip(x, target, method='translation', **kwargs):
    if method == 'translation':
        x_new = x + (target - x.mean())
    elif method == 'strech':
        multiplier = target / x.mean()
        x_new = multiplier * x
    elif method.__contains__('stretch_above'):
        base = kwargs['base']
        assert x.mean() >= base and target >= base
        multiplier = (target - base) / (x.mean() - base)
        x_new = base + multiplier * (x - base)
    elif method.__contains__('stretch_below'):
        base = kwargs['base']
        assert x.mean() <= base and target <= base
        multiplier = (base - target) / (base - x.mean())
        x_new = base - multiplier * (base - x)
    else:
        raise NotImplementedError
    assert np.abs(x_new.mean() - target) <= 1e-3, f'{x_new.mean()},{target}'
    if target > 1:
        assert x_new.min() > 1
    else:
        assert x_new.max() < 1
    return x_new


def get_clip_new(alpha, clip_max, clip_min, clipcontroltype):
    if clipcontroltype == 'average-clip':
        if alpha > 0:
            clip_max_new = 1 + alpha * (clip_max - 1)
            clip_max_new = normalize_clip(clip_max_new, target=clip_max.mean(), method='translation')

            clip_min_new = 1 - alpha * (1 - clip_min)
            clip_min_new = normalize_clip(clip_min_new, target=clip_min.mean(), method='translation')
        else:
            clip_max_new = 1 - alpha * 1. / (clip_max)
            clip_max_new = normalize_clip(clip_max_new, target=clip_max.mean(), method='stretch_above', base=1)
            clip_min_new = 1 + alpha * clip_min
            clip_min_new = normalize_clip(clip_min_new, target=clip_min.mean(), method='stretch_below', base=1)
            # TOD: 可以尝试不要归一化 1.2, 因为一般来说中间的采样非常多,归一化1.2以后中间的值很难说多大变化
    elif clipcontroltype == 'base-clip':
        if alpha >= 0:
            clip_max_new = clip_max.min() + alpha * (clip_max - clip_max.min())
            clip_min_new = clip_min.max() + alpha * (clip_min - clip_min.max())
        else:
            alpha = abs(alpha)
            clip_max_new = (clip_max.min() - 1) * 2 * 1. / (1 + np.exp(clip_max - clip_max.min()))
            clip_max_new = clip_max.min() - clip_max_new.max() + clip_max_new.max() + alpha * (
                    clip_max_new - clip_max_new.max())
            clip_min_new = 1. / clip_max_new
            # print( f'max:{clip_min_new.max()},min:{clip_min_new.min()}' )
            # clip_min_new = (1-clip_min.max()) * 2* 1./ (1+ np.exp(clip_min - clip_min.max()))
            # clip_min_new = clip_min.max() - clip_min_new.min() + clip_min_new.min() + alpha *( clip_min_new-clip_min_new.min() )
    else:
        raise NotImplementedError()

    return clip_max_new, clip_min_new


def prepare_data(dim, delta, sharelogsigma, clipcontroltype, cliprange, clip_clipratio, search_delta=False):
    global ress_tf_last
    path_data = path_root + '/KL2Clip/data/train_lambda'
    Name = f'dim={dim}, delta={delta}, train'
    path_data_processed = path_data + f'/{Name}'
    tools.mkdir(path_data_processed)

    if dim == 1:
        logsigma0s = np.array([0])
    else:
        raise NotImplementedError
    logsigma0s = logsigma0s.reshape((-1, dim))
    batch_size = 2048
    mu = np.zeros((dim,))

    opt = KL2Clip(dim=dim, batch_size=batch_size, sharelogsigma=sharelogsigma, clipcontroltype=clipcontroltype,
                  cliprange=cliprange)

    def get_fn_sample():
        mu0 = tf.placeholder(shape=[dim], dtype=tf.float32)
        a = tf.placeholder(shape=[batch_size, dim], dtype=tf.float32)
        logsigma0 = tf.placeholder(shape=[dim], dtype=tf.float32)
        sample_size = tf.placeholder(shape=(), dtype=tf.int32)
        dist = DiagGaussianPd(tf.concat((mu0, logsigma0), axis=0))
        samples = dist.sample(sample_size)
        fn_sample = U.function([mu0, logsigma0, sample_size], samples)
        fn_p = U.function([mu0, logsigma0, a], dist.p(a))
        return fn_sample, fn_p

    sess = U.make_session(make_default=True)
    results = []
    fn_sample, fn_p = get_fn_sample()
    for logsigma0 in logsigma0s:
        prefix_save = f'{path_data_processed}/logsigma0={logsigma0}'
        Name_f = f"{Name},logsigma0={logsigma0}"
        file_fig = f'{prefix_save}.png'
        # a_s_batch = fn_sample( mu, logsigma0, batch_size )
        a_s_batch = np.linspace(-5, 5, batch_size).reshape((-1, 1))
        logsigma0s_batch = np.tile(logsigma0, (batch_size, 1))
        print(a_s_batch.max(), a_s_batch.min())
        # --- sort the data: have problem in 2-dim
        # inds = np.argsort(a_s_batch, axis=0)
        # inds = inds.reshape(-1)
        # a_s_batch = a_s_batch[inds]
        # logsigma0s_batch = logsigma0s_batch[inds]

        # tools.reset_time()
        # a_s_batch.fill(0)
        # print(a_s_batch.shape)
        # a_s_batch[0, :]=0
        # if search_delta:
        # for i in range( batch_size):
        # a_s_batch[i,:] = 0.001 * (batch_size-i)
        if not os.path.exists(f'{prefix_save}.pkl'):
            # ress_tf = opt( mu0_logsigma0_tuple=(a_s_batch, logsigma0s_batch), a=None, delta=delta, clip_clipratio=clip_clipratio)
            ress_tf = opt(mu0_logsigma0_tuple=(np.zeros_like(logsigma0s_batch), logsigma0s_batch), a=a_s_batch,
                          delta=delta, clip_clipratio=clip_clipratio)
            print(a_s_batch[0], ress_tf.x.max[0], ress_tf.x.min[0])

            save_vars(f'{prefix_save}.pkl', a_s_batch, logsigma0, logsigma0s_batch, ress_tf)
        print(prefix_save)
        a_s_batch, logsigma0, logsigma0s_batch, ress_tf = load_vars(f'{prefix_save}.pkl')

        if search_delta:
            results.append(ress_tf)
            break
        if cliprange == clipranges[0]:  # TODO tmp
            fig = plt.figure(figsize=(20, 10))
        markers = ['^', '.']
        colors = [['blue', 'red'], ['green', 'hotpink']]
        # for ind, opt_name in enumerate(['max']):
        for ind, opt_name in enumerate(['max', 'min']):
            # if ind == 1:
            #     continue
            # --- plot tensorflow result
            ratios, cons = ress_tf.ratio[opt_name], ress_tf.con[opt_name]
            print(
                f'clip-{opt_name}_mean:{ratios.mean()}, clip-{opt_name}_min:{ratios.min()}, clip-{opt_name}_max:{ratios.max()}')
            if search_delta:
                continue
            if DEBUG:
                pass
            inds_good = cons <= get_ConstraintThreshold(ress_tf.delta)
            inds_bad = np.logical_not(inds_good)
            if dim == 1:
                if ind == 0 and 1:
                    ps = fn_p(mu, logsigma0, a_s_batch)
                    # +np.abs(ps.max()) + 1
                    ratio_new = -np.log(ps)
                    ratio_new = ratio_new - ratio_new.min() + ratios.min()
                    alpha = np.exp(-ps * 2)
                    print(alpha)
                    # plt.scatter(a_s_batch, ratio_new, s=5, label='ratio_new0')
                    ratio_new = ratio_new.min() + alpha * (ratio_new - ratio_new.min())
                    # plt.scatter( a_s_batch, ratio_new, s=5, label='ratio_new1' )

                    # ps = -ps
                    # ratios = ps - ps.min() + ratios.min()
                    # print( ps.min() )
                    # ratios_new =np.square( a_s_batch-mu ) * np.exp( -logsigma0 )
                    # ratio_min = ps  / (ps.max()-ps.min()) * ress_tf.ratio.min.max()
                    # plt.scatter(a_s_batch, ratio_min, s=5, label='square')
                    # plt.scatter(a_s_batch, 1./ratio_min, s=5, label='square')
                    # plt.scatter(a_s_batch, 1./ratios, s=5, label='1/max')

                def plot_new(alpha):
                    clip_max_new, clip_min_new = get_clip_new(alpha, ress_tf.ratio['max'], ress_tf.ratio['min'],
                                                              clipcontroltype=clipcontroltype)
                    plt.scatter(a_s_batch, clip_max_new, s=5, label=f'clip_max_{alpha}')
                    plt.scatter(a_s_batch, clip_min_new, s=5, label=f'clip_min_{alpha}')

                if ind == 0:
                    pass
                    # plot_new(0.5)
                    # plot_new(0.5)
                    # plot_new(-1)

                plt.scatter(a_s_batch[inds_good], ratios[inds_good], label='ratio_predict-good_' + opt_name, s=5,
                            color=colors[ind][0], marker=markers[ind])
                plt.scatter(a_s_batch[inds_bad], ratios[inds_bad], label='ratio_predict-bad_' + opt_name, s=5,
                            color=colors[ind][1], marker=markers[ind])
            elif dim == 2:
                ax = fig.gca(projection='3d')
                # ax.view_init(30, 30)
                ax.view_init(90, 90)
                # ax.plot_trisurf(a_s_batch[:, 0], a_s_batch[:, 1], ratios)
                ax.scatter(a_s_batch[inds_good, 0], a_s_batch[inds_good, 1], ratios[inds_good],
                           label='ratio_predict-good_' + opt_name, s=5, color=colors[ind][0], marker=markers[ind])
                ax.scatter(a_s_batch[inds_bad, 0], a_s_batch[inds_bad, 1], ratios[inds_bad],
                           label='ratio_predict-bad_' + opt_name, s=5, color=colors[ind][1], marker=markers[ind])

        if dim <= 2 and not search_delta:
            plt.title(
                Name_f + f'\nstep:{ress_tf.step},rate_satisfycon:{ress_tf.rate_satisfycon_}, rate_statisfydifference_:{ress_tf.rate_statisfydifference_}, difference_max_:{ress_tf.difference_max_}')
            plt.legend(loc='best')
            if not DEBUG:
                plt.savefig(file_fig)
    opt.close()
    if dim <= 2 and not search_delta:
        if DEBUG:
            if cliprange == clipranges[-1]:
                plt_tools.set_postion()
                plt.show()
    plt.close()


Delta_Default = None  # Default Constraint Threshold. 1-dim: 1.4e-2, 2-dim:0.0063
ConstraintThreshold_Multiplier = 1e-1
Rate_StatisfyConstraint = 0.99  # Default:0.99
Difference_Threshold = 0.01  # Default:0.01
Rate_Statisfydifference = 0.999  # Default: 0.999
Rate_StatisfyRatio = 1  # Default: 1
indBlock = 5  # Min update block. Default: 3 TOD tmp
DEBUG = 0


def f1(deltas):
    for d in deltas:
        d = np.round(d, 4)
        clipranges = [0.2]

        for cliprange in clipranges:
            Name = f'Adagrad,Delta_Default={Delta_Default},ConstraintThreshold_Multiplier={ConstraintThreshold_Multiplier},dim={dim}'
            prepare_data(dim=dim, delta=d, sharelogsigma=sharelogsigma, clipcontroltype=clipcontroltype,
                         cliprange=cliprange, clip_clipratio=clip_clipratio, search_delta=False)


if __name__ == '__main__':
    # search()
    # exit()
    # Current Setting
    clip_clipratio = None
    sharelogsigma = False
    clipcontroltype = 'base-clip'
    clipranges = [0.2]
    cliprange = 0.2
    # DEBUG = 1
    dim = 1
    '''
    for delta in np.arange(0.0002, 0.004, 0.0001):
        delta = np.round(delta, 4)
        clipranges = [0.2]

        for cliprange in clipranges:
            Name = f'Adagrad,Delta_Default={Delta_Default},ConstraintThreshold_Multiplier={ConstraintThreshold_Multiplier},dim={dim}'
            prepare_data(dim=dim, delta=delta, sharelogsigma=sharelogsigma, clipcontroltype=clipcontroltype,
                         cliprange=cliprange, clip_clipratio=clip_clipratio, search_delta=False)
    '''
    p = multiprocessing.Pool(4)
    f2 = lambda x: prepare_data(dim=dim, delta=np.round(x, 4), sharelogsigma=sharelogsigma,
                                clipcontroltype=clipcontroltype,
                                cliprange=0.2, clip_clipratio=clip_clipratio, search_delta=False)
    p.map(f2, np.arange(0.0002, 0.1, 0.0001))

    # LearningRates = [.01,0.02,0.05,0.1]
    # NumStepMaxs = [5000]
    # NumStepMins = [1000]
    # for LearningRate in LearningRates:
    #     # for DecayRate in DecayRates:
    #         for NumStepMin in NumStepMins:
    #             for NumStepMax in NumStepMaxs:
    #                 Name = f'LearningRate:{LearningRate},NumStepMin:{NumStepMin}, NumStepMax:{NumStepMax}'
    #                 view_result(dim=1, delta=Delta_Default)
    # tf_opt(batch_size=2048, dim=1, delta=.01)
    # opt = tfOpt(dim=1, batch_size=1)
    # ratio_max, ratio_min, con_max, con_min = opt( x0_batch=np.array([[1.4678326,-1.9064916]]) , a_batch=np.array([[1.2847137]]) )
    # print(ratio_max, ratio_min)
    # exit()

    # main(batch_size=1, dim=1)

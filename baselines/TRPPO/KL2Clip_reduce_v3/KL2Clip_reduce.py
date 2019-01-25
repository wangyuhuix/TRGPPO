import pathos.multiprocessing as multiprocessing

from baselines.common.tools import load_vars, save_vars
from dotmap import DotMap

import os
import time

import numpy as np
from baselines.common import tools
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

from baselines.TRPPO.KL2Clip_reduce_v3.KL2Clip_NN_normal import KL2Clip_NN, KL2Clip_tabular, \
    TabularActionPrecision
from baselines.common.distributions import DiagGaussianPd

import tensorflow as tf
from tensorflow.contrib.constrained_optimization import ConstrainedMinimizationProblem, AdditiveExternalRegretOptimizer


AVERAGE_CLIP = 'average-clip'
BASE_CLIP = 'base-clip'
NONE_CLIP = 'none-clip'


def arr2meshgrid(arr, dim):
    meshgrids = [arr for i in range(dim)]
    return multiarr2meshgrid(meshgrids)


def multiarr2meshgrid(meshgrids):
    arr = np.meshgrid(*meshgrids)
    arr = [i.reshape(-1, 1) for i in arr]
    arr = np.concatenate(arr, axis=1)
    return arr


class KL2Clip(object):
    def __init__(self, dim, batch_size=None, use_tabular='True',sharelogsigma=False):
        if use_tabular:
            opt1Dkind = 'tabular'
        else:
            opt1Dkind = 'nn'
        print(f'you are using kl2clip_reduce_v3 with opt1Dkind={opt1Dkind}')
        self.dim = dim
        self.sharelogsigma = sharelogsigma
        if opt1Dkind == 'tabular':
            self.opt1D = KL2Clip_tabular()
        elif opt1Dkind == 'nn':
            self.opt1D = KL2Clip_NN()
            self.tes()  # TODO: debug
        else:
            raise NotImplementedError('Unknown opt1Dkind, please use tabular or nn')

    def tes(self):
        batch_size = 1
        dim = self.dim
        cliprange = 0.2
        ress = self(mu0_logsigma0_tuple=(np.zeros((batch_size, dim), dtype=np.float32), np.zeros((batch_size, dim))),
                    a=np.zeros((batch_size, dim)), clipcontroltype=BASE_CLIP, cliprange=cliprange, silent=True)
        precision = np.abs(ress.ratio.max[0] - cliprange - 1)
        assert precision <= 1e-4, f'Please check the model of KL2Clip_NN. Precision:{precision}'
        print('KL2Clip Pass Testing! Precision=', np.abs(ress.ratio.max[0] - cliprange - 1))

    def get_cliprange_by_delta(self, delta):
        batch_size = 1
        dim = self.dim
        ress = self(mu0_logsigma0_tuple=(np.zeros((batch_size, dim), dtype=np.float32), np.zeros((batch_size, dim))),
                    a=np.zeros((batch_size, dim)), clipcontroltype=NONE_CLIP, delta=delta, silent=True)
        return ress.ratio.max[0] - 1

    def __call__(self,
                 mu0_logsigma0_cat=None, mu0_logsigma0_tuple=None, a=None,
                 delta=None, clipcontroltype=None, cliprange=None,
                 sharelogsigma=False, clip_clipratio=None,
                 silent=False, pas=None):  # TODO 修改相关调用函数
        '''
        delta is not None: compute with delta
        delta is None: compute delta with (clipcontroltype, cliprange)
        :param mu0_logsigma0_cat:
        :type mu0_logsigma0_cat:
        :param mu0_logsigma0_tuple:
        :type mu0_logsigma0_tuple:
        :param delta:
        :type delta:
        :param a:
        :type a:
        :param clip_clipratio:
        :type clip_clipratio:
        :param clipcontroltype:
        :type clipcontroltype:
        :param cliprange:
        :type cliprange:
        :param silent:
        :type silent:
        :return:
        :rtype:
        '''
        dim = self.dim
        assert not sharelogsigma
        if delta is None:
            if clipcontroltype == BASE_CLIP:
                if callable(cliprange):
                    cliprange = cliprange(None)
                target = 1 + cliprange
                logsigma = -np.log(target) * 2 / dim
                delta = dim * (-logsigma + np.exp(logsigma) - 1) / 2
                if not silent:
                    print(
                        f'The given delta is None, we set delta={delta} by  clipcontroltype={clipcontroltype}, cliprange={cliprange}, dim={dim},sharelogsigma={sharelogsigma}')
            else:
                raise NotImplementedError
        # print(f'delta={delta}')
        if mu0_logsigma0_tuple is not None:
            mu0, logsigma0 = mu0_logsigma0_tuple
            mu0_logsigma0_cat = np.concatenate((mu0, logsigma0), axis=-1)
        elif mu0_logsigma0_cat is not None:
            mu0 = mu0_logsigma0_cat[:, :dim]
            logsigma0 = mu0_logsigma0_cat[:, -dim:]
        else:
            raise Exception('You must supply (mu0 logsigma0)!')

        if a is None:
            a = np.zeros((mu0_logsigma0_cat.shape[0], dim), dtype=np.float32)

        assert delta is not None
        assert mu0_logsigma0_cat.shape[1] // 2 == dim

        sigma0 = np.exp(logsigma0)
        z = (mu0 - a) / sigma0
        a = np.linalg.norm(z, axis=1) / np.sqrt(self.dim)  # shape (batch_size, 1)
        # ratio_min, ratio_max, lambda_min_nnoutput, lambda_max_nnoutput, \
        # lambda_min_scipyfsolve, lambda_max_scipyfsolve, mu_logsigma_min, mu_logsigma_max = \
        #     self.nn(action=a, delta=delta / self.dim, USE_MULTIPROCESSING=False)  # 1d mu, sigma

        # TODO: 这一行可以用tabular形式解决
        # ratio_min, ratio_max, min_mu_logsigma_fsolve, max_mu_logsigma_fsolve = self.nn(action=a, delta=delta / self.dim,
        #                                                                                USE_MULTIPROCESSING=False)
        ratio_min, ratio_max, min_mu_logsigma_fsolve, max_mu_logsigma_fsolve = self.opt1D(action=a,
                                                                                          delta=delta / self.dim, )
        ratio_min = np.power(ratio_min, self.dim)
        ratio_max = np.power(ratio_max, self.dim)

        if clip_clipratio is not None:
            assert clip_clipratio >= 1
            clip_clipratio_new = ratio_max.min() - 1 + clip_clipratio
            print(f'clip clipratio to {clip_clipratio_new}')

            ratio_max = np.minimum(ratio_max, clip_clipratio_new)
            ratio_min = np.maximum(ratio_min, 1. / clip_clipratio_new)

        return DotMap(
            ratio=DotMap(max=ratio_max, min=ratio_min),
            mu_logsigma=DotMap(max=max_mu_logsigma_fsolve, min=min_mu_logsigma_fsolve),
            delta=delta
        )


def tes_3d_data():
    if tools.ispc('xiaoming'):
        path_root = '/media/root/新加卷/KL2Clip'
    else:
        path_root = ''
    import plt_tools
    from baselines.common.tools import load_vars, save_vars
    import matplotlib.pyplot as plt
    if 1:
        dim = 1
        # tf.logging.set_verbosity(tf.logging.INFO)
        files = []

        path_data = f'{path_root}/data/train'
        for dir in sorted(os.listdir(path_data)):
            dir_pickle = os.path.join(path_data, dir)
            try:
                file_path = os.listdir(dir_pickle)[0] if os.listdir(dir_pickle)[0].endswith('pkl') else \
                    os.listdir(dir_pickle)[1]
            except:
                continue

            files.append(os.path.join(dir_pickle, file_path))

        tfoptsssss = []
        scipyfsolvesssss = []
        a_delta = []
        # exit()
        # files = ['/media/root/新加卷/KL2Clip/data/train/dim=1, delta=0.0902, train/logsigma0=[0].pkl']
        for ind, f in enumerate(files):  # enumerate(files[1::100]):
            print(f)
            actions, _, _, ress_tf = load_vars(f)
            delta = ress_tf.delta
            # min_mu_logsigma = ress_tf.x.min
            # max_mu_logsigma = ress_tf.x.max
            ratio_min_tfopt, ratio_max_tfopt = ress_tf.ratio.min, ress_tf.ratio.max

            kl2clip = KL2Clip(dim=dim)
            x0 = np.zeros(shape=(actions.shape[0], 2), dtype=np.float32)

            # sort by actions
            inds = np.argsort(actions, axis=0)
            inds = inds.reshape(-1)

            actions = actions[inds]
            ratio_min_tfopt, ratio_max_tfopt = ratio_min_tfopt[inds], ratio_max_tfopt[inds]
            ress = kl2clip(mu0_logsigma0_cat=x0, a=actions, delta=delta)
            ratio_min_scipyfsolve, ratio_max_scipyfsolve = ress.ratio.min, ress.ratio.max
            a_delta.append(np.concatenate((actions, delta * np.ones_like(actions)), axis=1))
            tfoptsssss.append(ratio_max_tfopt)
            scipyfsolvesssss.append(ratio_max_scipyfsolve)

        save_vars('aa.pkl', a_delta, tfoptsssss, scipyfsolvesssss)

    a_delta, tfoptsssss, scipyfsolvesssss = load_vars('aa.pkl')

    def filter(arr):
        for ind in range(len(arr)):
            arr[ind] = arr[ind][0::30]
        return arr

    a_delta, tfoptsssss, scipyfsolvesssss = [filter(item) for item in (a_delta, tfoptsssss, scipyfsolvesssss)]
    a_delta = np.concatenate(a_delta, axis=0)
    tfoptsssss = np.concatenate(tfoptsssss, axis=0)
    scipyfsolvesssss = np.concatenate(scipyfsolvesssss, axis=0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(0, 0)
    ax.scatter(a_delta[:, 0], a_delta[:, 1], tfoptsssss, '_tfopt', s=1, color='black')
    ax.scatter(a_delta[:, 0], a_delta[:, 1], scipyfsolvesssss, '_scipyfsolve', s=1, color='red')
    plt_tools.set_postion()
    plt_tools.set_size()
    # plt_tools.set_equal()
    plt.show()


import baselines.common.tf_util as U


def get_func_cons(batch_size):
    mu_logsigma = tf.placeholder(shape=[batch_size, 2], dtype=tf.float32)
    delta = tf.placeholder(shape=(1,), dtype=tf.float32)
    x0 = tf.zeros(shape=[batch_size, 2])
    distNormal = DiagGaussianPd(x0)
    dist = DiagGaussianPd(mu_logsigma)
    kl = dist.kl(distNormal)
    cons = kl - delta
    fn_get_cons = U.function([mu_logsigma, delta], cons)
    return fn_get_cons


def tes_data():
    from baselines.common.tools import load_vars, save_vars
    import matplotlib.pyplot as plt
    dim = 1
    path_root = '/home/hugo/Desktop/wxm/KL2Clip'
    path_data = f'{path_root}/data/train'
    # files = tools.get_files(path_rel=f'/home/hugo/Desktop/wxm/KL2Clip/data/dim={dim}, delta=0.01')
    # files = ['/root/d/e/et/baselines/KL2Clip/data/dim=2, delta=0.01/logsigma0=[0. 0.].pkl']  # TODO tmp
    # actions, deltas, max_mu_logsigma, min_mu_logsigma = [], [], [], []

    # path_data = '/home/hugo/Desktop/wxm/KL2Clip/data/train/dim=1, delta=0.027, train'
    files = tools.get_files(path_rel=path_data, only_sub=False, sort=False, suffix='.pkl')
    kl2clip_tabular = KL2Clip(dim=dim, opt1Dkind='tabular')
    kl2clip_nn = KL2Clip(dim=dim, opt1Dkind='nn')

    for ind, f in enumerate(files[:]):
        actions, logsigma0, logsigma0s_batch, ress_tf = load_vars(f)
        actions = np.round(actions, TabularActionPrecision)
        # actions = np.float32(actions)
        # logsigma0 = np.float32(logsigma0)
        # logsigma0s_batch = np.float32(logsigma0s_batch)

        delta = ress_tf.delta
        x0 = np.concatenate((np.zeros_like(logsigma0s_batch), logsigma0s_batch), axis=1)

        # x0 = np.float32(x0)
        # actions = np.float32(actions)
        # delta = np.float32(delta)

        ress_tabular = kl2clip_tabular(mu0_logsigma0_cat=x0, a=actions, delta=delta)
        ress_nn = kl2clip_nn(mu0_logsigma0_cat=x0, a=actions, delta=delta)
        # exit()

        # time1 = time.time()
        # print('time:', time1- time0)
        # print(actions.shape)
        # exit()
        # ratio_min_scipyfsolve, ratio_max_scipyfsolve = ress.ratio.min, ress.ratio.max
        # lambda_scipyfsolve = ress.lambda_scipyfsolve
        ratios_tfopt = ress_tf.ratio
        ratios_tabular = ress_tabular.ratio
        ratios_nn = ress_nn.ratio

        print(f'ress_tabular is {ress_tabular}')
        print(f'ress_nn is {ress_nn}')
        keys_ress = ['min', 'max']
        name_base = f'delta: {delta}, logsigma0: {logsigma0}, delta x 100'

        fig = plt.figure(figsize=(20, 10))
        if dim == 1:
            for minORmax in keys_ress:  # for maximize and minimize

                # print('cons_scipy.fsolve: ', cons_final)
                # print('cons_tf.opt: ', ress_tf.con[minORmax])
                # plt.scatter(actions, ratios_scipyfsolve[minORmax], label='ratio_' + minORmax + '_scipyfsolve', color='blue',
                #             s=1)
                '''
                mu_logsigma = ress.mu_logsigma[minORmax]
                cons_func = get_func_cons(batch_size=mu_logsigma.shape[0])
                cons_tfopt = ress_tf.con[minORmax]
                cons_final = cons_func(mu_logsigma, np.array([delta], np.float32))
                threshold = 1e-7
                plt.scatter(actions[cons_tfopt < threshold], ratios_tfopt[minORmax][cons_tfopt < threshold],
                            label='ratio_' + minORmax + '_tfopt-good', color='black', s=1)
                plt.scatter(actions[cons_tfopt >= threshold], ratios_tfopt[minORmax][cons_tfopt >= threshold],
                            label='ratio_' + minORmax + '_tfopt-bad', color='pink', s=1)

                plt.scatter(actions[cons_final < threshold], ratios_scipyfsolve[minORmax][cons_final < threshold],
                            label='ratio_' + minORmax + '_scipyfsolve-good', color='blue', s=1)
                plt.scatter(actions[cons_final >= threshold], ratios_scipyfsolve[minORmax][cons_final >= threshold],
                            label='ratio_' + minORmax + '_scipyfsolve-bad', color='red', s=1)
                
                '''
                plt.scatter(actions, ratios_tfopt[minORmax], label='ratio_' + minORmax + '_tfopt', color='black', s=1)
                plt.scatter(actions, ratios_tabular[minORmax], label='ratio_' + minORmax + '_tabular', color='red',
                            s=1)
                plt.scatter(actions, ratios_nn[minORmax], label='ratio_' + minORmax + '_nn', color='green',
                            s=1)


        elif dim == 2:
            ax = fig.gca(projection='3d')
            print(actions.shape, ratios_tfopt['max'].shape, ratios_tfopt['min'].shape)
            for opt_name in keys_ress:  # for maximize and minimize

                ax.scatter(actions[:, 0], actions[:, 1], ratios_tfopt[opt_name],
                           label='ratio_' + opt_name + '_tfopt', color='blue', s=1)
                ax.scatter(actions[:, 0], actions[:, 1], ratios_scipyfsolve[opt_name],
                           label='ratio_' + opt_name + '_scipyfsolve', color='red', s=1)

        name = name_base + ', ratio'
        plt.title(name)
        plt.legend(loc='best')

        path_dir, _ = os.path.split(f)
        # plt.savefig(path_dir + f'/{name}.png')
        print('save' + path_dir + f'/delta:{delta}.png' + '   ratio')
        plt.show()
        # plt.close()

        # exit()

    # plt.show()


def tes_sample(dim, delta=None, cliprange=None):
    # from baselines.TRPPO.KL2Clip.KL2Clip_opt_tf import KL2Clip as KL2Clip_tfopt
    # from baselines.TRPPO.Kl2Clip_reduce_v1.KL2Clip_reduce import KL2Clip as KL2Clip_reduce_v1
    batch_size = 2048
    print('asdasdasd')

    if 1:
        sample_size = int(np.power(batch_size, 1. / dim))
        batch_size = np.power(sample_size, dim)
        a = np.linspace(-3.5, 3.5, sample_size)
        a = arr2meshgrid(a, dim)
        mu = np.zeros((batch_size, dim), dtype=np.float32)
        logsigma = np.zeros((batch_size, dim))
    else:
        mu = np.zeros((batch_size, dim), dtype=np.float32)
        logsigma = np.zeros((batch_size, dim))
        import baselines.TRPPO.KL2Clip.tools_ as tools_
        fn_sample, _ = tools_.get_fn_sample_fn_p(dim)
        sess = U.make_session()
        with sess:
            a = fn_sample(mu[0], logsigma[0], batch_size)
        a[0, :] = 0

    kl2clip_tabular = KL2Clip(dim=dim, opt1Dkind='tabular')
    kl2clip_nn = KL2Clip(dim=dim, opt1Dkind='nn')

    kl2clip = KL2Clip(dim=dim, )
    # kl2clip_reduce_v1 = KL2Clip_reduce_v1(dim=dim, )
    # kl2clip_tfopt = KL2Clip_tfopt(dim=dim, batch_size=batch_size, sharelogsigma=False, clipcontroltype=BASE_CLIP,cliprange=0.2)

    # fs = [kl2clip, kl2clip_reduce_v1, kl2clip_tfopt]
    fs = [kl2clip]
    num_fs = len(fs)
    colors = ['black', 'blue', 'green']
    markers = ['.', '^', '*']
    # names = ['reduce_v3', 'reduce_v1', 'tfopt']
    names = ['reduce_v3', 'tfopt']
    ress_ss = []
    for ind, f in enumerate(fs):
        if ind == 0:
            ress = f(mu0_logsigma0_tuple=(mu, logsigma), a=a, delta=delta, clipcontroltype=BASE_CLIP,
                     cliprange=cliprange)
        else:
            ress = f(mu0_logsigma0_tuple=(mu, logsigma), a=a, delta=delta)
        ress_ss.append(ress)
    fig = plt.figure(figsize=(20, 10))
    print(f'dim={dim}, delta={delta}')
    names_col = ' ,max,min,mean,median,ratio_a0,delta'
    names_col = names_col.split(',')
    for name in names_col:
        print(f"|{name: ^12}", end='')
    print('|')
    for name in names_col:
        print(f"|{' ':-<12}", end='')
    print('|')
    for i in range(num_fs):
        res = ress_ss[i]
        name = names[i]

        values = [name, res.ratio.max.max(), res.ratio.max.min(), res.ratio.max.mean(), np.median(res.ratio.max),
                  res.ratio.max[0], res.delta]
        for v in values:
            if isinstance(v, str):
                print(f'|{v: <12}', end='')
            else:
                print(f'|{v:12.8f}', end='')
        print('|')
    if dim <= 2:
        if dim == 2:
            ax = fig.gca(projection='3d')
            ax.view_init(30, 30)
        for i in range(num_fs):
            res = ress_ss[i]
            name = names[i]
            if dim == 1:

                plt.scatter(a[:, 0], res.ratio.max, linewidth=0.5, color=colors[i], alpha=0.5, label=name, s=10,
                            marker=markers[i])
                plt.scatter(a[:, 0], res.ratio.min, linewidth=0.5, color=colors[i], alpha=0.5, label=name, s=10,
                            marker=markers[i])
            elif dim == 2:
                ax.plot_trisurf(a[:, 0], a[:, 1], res.ratio.max, linewidth=0.5, color=colors[i], alpha=0.5)
                ax.plot_trisurf(a[:, 0], a[:, 1], res.ratio.min, linewidth=0.5, color=colors[i], alpha=0.5)
        plt.legend()
        plt.show()


def tes():
    batch_size = 1
    dims = [1]
    # clipranges = [0.1,0.2,0.3]
    clipranges = [0.2]
    delta = 0.02954377979040146
    action = 0.00020964576106052846
    # action = 0.000

    print(clipranges)
    for dim in dims:
        for cliprange in clipranges:
            kl2clip = KL2Clip(dim=dim, batch_size=1)
            temp = kl2clip(mu0_logsigma0_cat=np.array([[0., 0.], ]), a=action, delta=delta)
            print(temp)


if __name__ == '__main__':
    # kl2clip = KL2Clip(dim=1, batch_size=1)
    # action = [0]
    # kl2clip(mu0_logsigma0_cat=np.array([[0., 0.], ]), a=action, cliprange=0.2, clipcontroltype=BASE_CLIP)
    # exit()
    dims = [1]
    cliprange = [0.2]
    deltas = [0.03]
    for dim in dims:
        for delta in deltas:
            tes_sample(dim, delta)
            ops.reset_default_graph()
    exit()
    # kl2clip = KL2Clip(0.3, )

    tes_data()
    # tes()
    # tes_sample(dim=1, delta=None, cliprange=0.2)

from scipy.optimize import fsolve

from dotmap import DotMap

import os
import time

import numpy as np
from baselines.common import tools
import pandas as pd

from baselines.common import plt_tools

from math import log, exp

AVERAGE_CLIP = 'average-clip'
BASE_CLIP = 'base-clip'
NONE_CLIP = 'none-clip'
TabularActionPrecision = 4
# path_root = '../../KL2Clip'
path_root = os.path.abspath('./KL2Clip')

class KL2Clip(object):
    def __init__(self, dim, batch_size=None, sharelogsigma=False, use_tabular='True'):
        if use_tabular:
            opt1Dkind = 'tabular'
        else:
            opt1Dkind = 'fsolve'
        print(f'You are using KL2Clip_{opt1Dkind}.')
        self.dim = dim
        self.sharelogsigma = sharelogsigma
        if opt1Dkind == 'tabular':
            self.opt1D = KL2Clip_tabular()
        elif opt1Dkind == 'fsolve':
            self.opt1D = KL2Clip_fsolve()
        else:
            raise NotImplementedError('Unknown opt1Dkind, please use tabular or nn')

    def get_cliprange_by_delta(self, delta):
        raise NotImplementedError

    def __call__(self,
                 mu0_logsigma0_cat=None, mu0_logsigma0_tuple=None, a=None,
                 delta=None, clipcontroltype=None, cliprange=None,
                 sharelogsigma=False, clip_clipratio=None,
                 silent=False, pas=None):  # TODO 修改相关调用函数
        dim = self.dim
        assert not sharelogsigma
        # print(f'delta={delta}')

        logits = mu0_logsigma0_cat
        ratio = self.opt1D(pas=pas, delta=delta)
        # TODO: 这里的处理方式需要斟酌
        ratio.max[np.isnan(ratio.max)] = np.nanmin(ratio.max)
        ratio.min[np.isnan(ratio.min)] = np.nanmax(ratio.min)
        # print(f'ratio.min is between:  [{ratio.min.min()}, {ratio.min.max()}]')
        # print(f'ratio.min.median:  {np.median(ratio.min)}  ratio.min.mean:  {np.mean(ratio.min)}')

        # print(f'ratio.max is between:  [{ratio.max.min()}, {ratio.max.max()}]')
        # print(f'ratio.max.median:  {np.median(ratio.max)}  ratio.max.mean:  {np.mean(ratio.max)}')
        # print(ratio)
        return DotMap(
            ratio=ratio,
            delta=delta
        )


class KL2Clip_fsolve(object):

    def f_setting1(self, pa, delta):
        def f(m):
            m = float(m)
            if (1 + m * pa) == 0:
                return None
            part_1 = (m ** (1 - pa))
            part_2 = ((1 / pa + m) ** pa)
            part_3 = ((1 - pa) / m + pa ** 2 / (1 + m * pa))
            part_1_2 = part_1 * part_2
            # return [part_1_2, part_3]
            if isinstance(part_1_2, complex):
                if abs(part_1_2.imag) > 1e-10:
                    return None
                part_1_2 = part_1_2.real
            part_1_2_3 = part_1_2 * part_3
            if part_1_2_3 > 0:
                return log(part_1_2_3) - delta
            return None

        return f

    def f_setting2(self, pa, delta):
        def f(r):
            if r < 0 \
                    or ((1 - pa) / (1 - pa * r)) < 0:
                return None
            return (1 - pa) * log((1 - pa) / (1 - pa * r)) - pa * log(r) - delta

        return f

    def opt_entity1(self, pa, delta, type='max', sol_ini=None):
        pa, delta = (float(item) for item in (pa, delta))
        f = self.f_setting1(pa, delta)
        if type == 'min':
            if sol_ini is not None:
                m_ini = sol_ini
            else:
                m_ini = 1
        else:
            if sol_ini is not None:
                m_ini = sol_ini
            else:
                m_ini = -1
                while f(m_ini) is None:
                    m_ini -= 1
        m = fsolve(f, m_ini, full_output=0)

        # lam  = exp(log(abs((m ** p0) * ((1 / p1 + m) ** p1))) - delta)
        # if m < 0:
        #     lam = -lam
        if isinstance(m, np.ndarray) and m.ndim >= 1:
            m = m[0]
        m = float(m)
        lam = m ** (1 - pa) * (1 / pa + m) ** pa / exp(delta)
        if isinstance(lam, complex):
            lam = lam.real
        qi_sum = lam * (1 - pa) / m
        qa = lam * pa / (1 / pa + m)
        ratio = qa / pa
        # print(f'type:{type},ratio:{ratio},qa:{qa},kl_constraint:{f(m)},m:{m},lam:{lam}')
        return ratio, m

    def opt_entity2(self, pa, delta, type='max', sol_ini=None):
        pa, delta = (float(item) for item in (pa, delta))
        f = self.f_setting2(pa, delta)
        if type == 'min':
            if sol_ini is not None:
                r_ini = sol_ini
            else:
                r_ini = 1. / exp(0.011 * (1. / pa))
        else:
            if sol_ini is not None:
                r_ini = sol_ini
            else:
                r_ini = exp(0.011 * (1. / pa))
                while f(r_ini) is None:
                    r_ini -= (r_ini - 1) * 0.9
        r = fsolve(f, r_ini, full_output=0)

        if isinstance(r, np.ndarray) and r.ndim >= 1:
            r = r[0]
        return r

    def __call__(self, pas, delta, initialwithpresol=False):
        ratio_pre = None
        sol_pre = None
        ratio_maxs = []
        for pa in pas:

            if pa <= 0.95:
                ratio, sol_pre = self.opt_entity1(pa, delta, 'max', sol_pre if initialwithpresol else None)
                ratio_maxs.append(ratio)
                ratio_pre = ratio
            else:
                ratio_pre = ratio = self.opt_entity2(pa, delta, 'max', ratio_pre if initialwithpresol else None)
                ratio_maxs.append(ratio)

            # print(f'pa:  {pa}   ratio_max:  {ratio}')

        ratio_pre = None
        sol_pre = None
        ratio_mins = []
        for pa in pas:
            # print(f'pa:  {pa}   ratio_min:  {ratio}')
            if pa <= 0.95:
                ratio, sol_pre = self.opt_entity1(pa, delta, 'min', sol_pre if initialwithpresol else None)
                ratio_mins.append(ratio)
                ratio_pre = ratio
            else:
                ratio_pre = ratio = self.opt_entity2(pa, delta, 'min', ratio_pre if initialwithpresol else None)
                ratio_mins.append(ratio)

        ratio_maxs = np.array(ratio_maxs)
        ratio_mins = np.array(ratio_mins)
        return DotMap(max=ratio_maxs, min=ratio_mins)


import tools_process

path_root_tabular = f'{path_root}/tabular'
tools.mkdir(path_root_tabular)
path_root_tabluar_locker = f'{path_root_tabular}/locker'
tools.mkdir(path_root_tabluar_locker)


class KL2Clip_tabular(object):
    def __init__(self):
        self.deltas_dict = {}
        self._upperbound = 0.99
        self._lowerbound = 0.01

    def get_tabular(self, delta):
        save_path = f'{path_root_tabular}/{delta:.16f}_atari'
        if delta in self.deltas_dict:
            pass
        # TODO: file lock
        elif os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            self.deltas_dict[delta] = tools.load_vars(save_path)
        else:
            with tools_process.FileLocker(f'{path_root_tabluar_locker}/{delta:.16f}'):
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    self.deltas_dict[delta] = tools.load_vars(save_path)
                else:
                    self.deltas_dict[delta] = self.create_tabular(delta)
                    tools.save_vars(save_path, self.deltas_dict[delta])
        return self.deltas_dict[delta]

    def create_tabular(self, delta):
        time0 = time.time()
        print(f'start to generate tabular of delta: {delta} Taburlar action precision{10 ** (-TabularActionPrecision)}')
        fsolver = KL2Clip_fsolve()
        pas = np.arange(self._lowerbound, self._upperbound + 10 ** (-TabularActionPrecision),
                        10 ** (-TabularActionPrecision))
        ratio_dicts = fsolver(pas=pas, delta=delta, initialwithpresol=True)
        ratio_min, ratio_max = ratio_dicts.min, ratio_dicts.max
        # 注意这里一定要指定actions为float32 不然默认float64 后面用Dataframe索引时会有bug
        tabular = {np.float64(pas[i].__round__(TabularActionPrecision)): (ratio_min[i], ratio_max[i]) for i in
                   range(pas.shape[0])}
        df = pd.DataFrame(data=tabular, dtype=np.float32)
        time0 = time.time() - time0
        print(f'Successfully generate tabular, with time {time0}s')
        return df

    def __call__(self, pas, delta):
        df = self.get_tabular(delta=delta)
        assert pas.ndim == 1
        pas = np.clip(pas, self._lowerbound, self._upperbound).astype(np.float64)
        pas = np.round(pas, TabularActionPrecision)
        # print('df.head:\n', df.head())
        # print('action.round(TabularActionPrecision):\n', action)

        # TODO: check df columns
        if len(str(df.columns[0])) > 7 or len(str(df.columns[1])) > 7:
            df.columns = np.round(df.columns.values.astype(np.float64), TabularActionPrecision)

        ratio_min, ratio_max = np.split(df.loc[:, pas].values, 2, axis=0)
        # ratio_min, ratio_max = np.split(df.reindex(columns=action).values, 2, axis=0)
        ratio_min, ratio_max = np.squeeze(ratio_min, 0), np.squeeze(ratio_max, 0)
        return DotMap(max=ratio_max, min=ratio_min)

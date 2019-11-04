# import pathos.multiprocessing as multiprocessing

from dotmap import DotMap

import baselines.common.tf_util as U

import os
from numpy import dot
import scipy
import time

import numpy as np
from toolsm import tools
# import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from baselines.ppo2_AdaClip.KL2Clip_reduce_v3.KL2Clip_NN_normal import KL2Clip_NN, KL2Clip_tabular, \
    TabularActionPrecision
from baselines.common.distributions import DiagGaussianPd

import tensorflow as tf




from enum import Enum


# TODO: 修改clipcontroltype类型
class Adjust_Type(Enum):
    origin = 0
    base_clip_upper = 1
    base_clip_lower = 2

class KL2Clip(object):
    def __init__(self, opt1Dkind='tabular', **kwargs):
        # print('**********************'+opt1Dkind)
        print(f'You are using kl2clip_reduce_v3 with opt1Dkind={opt1Dkind}')
        if opt1Dkind == 'tabular':
            self.opt1D = KL2Clip_tabular()
        elif opt1Dkind == 'nn':
            self.opt1D = KL2Clip_NN()
        else:
            raise NotImplementedError('Unknown opt1Dkind, please use tabular or nn')



    def tes_atzero(self):
        batch_size = 1
        dim = 1
        # cliprange = 0.2
        # ress = self(mu0_logstd0=(np.zeros((batch_size, dim), dtype=np.float32), np.zeros((batch_size, dim))),
        #             a=np.zeros((batch_size, dim)), adjusttype=Adjust_Type.base_clip_lower, cliprange=cliprange, silent=True)
        # print(ress.ratio.min[0])


        # cliprange = 1 - 1./1.2
        # ress = self(mu0_logstd0=(np.zeros((batch_size, dim), dtype=np.float32), np.zeros((batch_size, dim))),
        #             a=np.zeros((batch_size, dim)), adjusttype=Adjust_Type.base_clip_upper, cliprange=cliprange, silent=True)
        # print( ress.delta )
        # exit()
        precision = np.abs(ress.ratio.min[0] - (1-cliprange) )
        assert precision <= 1e-4, f'Please check the model of KL2Clip_NN. Precision:{precision}'
        print('KL2Clip Pass Testing! Precision=', precision)

    def delta2cliprange(self, delta, dim, adjusttype=Adjust_Type.base_clip_lower):
        if isinstance(adjusttype, str):
            adjusttype = Adjust_Type[adjusttype]
        batch_size = 1
        ress = self(mu0_logstd0=(np.zeros((batch_size, dim), dtype=np.float32), np.zeros((batch_size, dim))),
                    a=np.zeros((batch_size, dim)), delta=delta, silent=True)
        if adjusttype == Adjust_Type.base_clip_upper:
            return ress.ratio.max[0] - 1
        elif adjusttype == Adjust_Type.base_clip_lower:
            return 1 - ress.ratio.min[0]
        else:
            raise NotImplementedError

    def cliprange2delta(self, cliprange, dim, adjusttype=Adjust_Type.base_clip_lower):
        # Adjust delta by making the corresponding the lowest upper (or largest lower) clipping range equal to the specified clipping range.
        # Please see the clipping range result of TRGPPO for Gaussian policy in continuous action space.
        if isinstance(adjusttype, str):
            adjusttype = Adjust_Type[adjusttype]
        assert cliprange is not None
        assert not callable(cliprange)
        if adjusttype == Adjust_Type.base_clip_upper:
            target = 1 + cliprange # decide delta depending on lowest upper cliprange
        elif adjusttype == Adjust_Type.base_clip_lower:
            target = 1 - cliprange # decide delta depending on largest lower cliprange
        else:
            raise NotImplementedError
        logstd = -np.log(target) * 2 / dim
        delta = dim * (-logstd + np.exp(logstd) - 1) / 2
        return delta

    def __call__(self,
                 mu0_logstd0, a=None,
                 delta=None,
                 cliprange=None, adjusttype=Adjust_Type.origin,
                 require_sol=False,
                 # sharelogstd=False, clip_clipratio=None,
                 verbose=True, **kwargs):  # TODO 修改相关调用函数
        '''
        If delta is provided, then it's used for kl2clip;
        otherwise, you should at least provide cliprange, and it is used to calculate the delta by the rule of <adjustdelta_type>.
        :param mu0_logstd0:
        :param a:
        :param delta:
        :type delta:
        :param cliprange:
        :type cliprange:
        :param adjusttype:
        :type adjusttype:
        :param require_sol:
        :type require_sol:
        :param sharelogstd:
        :type sharelogstd:
        :param clip_clipratio:
        :type clip_clipratio:
        :param verbose:
        :type verbose:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        '''
        # ---  Preprocessing Args
        # assert not sharelogstd, 'Not Implemented'


        if isinstance( mu0_logstd0, tuple ):
            mu0, logstd0 = mu0_logstd0
            dim = mu0.shape[-1]
            mu0_logstd0 = np.concatenate((mu0, logstd0), axis=-1)
        else:
            if isinstance(mu0_logstd0, list):
                mu0_logstd0 = np.array(mu0_logstd0)
            assert isinstance( mu0_logstd0, np.ndarray )
            dim = mu0_logstd0.shape[-1] // 2
            mu0 = mu0_logstd0[:, :dim]
            logstd0 = mu0_logstd0[:, -dim:]

        if a is None:
            a = np.zeros((mu0_logstd0.shape[0], dim), dtype=np.float32)

        # assert (delta is not None) != (cliprange is not None), 'You should provide either one of these two values'

        #------ Adjust delta by specified cliprange
        if isinstance(adjusttype, str):
            adjusttype = Adjust_Type[adjusttype]
        # TODO: 根据adjusttype来判断应该用什么
        if delta is None:
            delta = self.cliprange2delta( cliprange, dim, adjusttype )
            if not hasattr(self,'delta') or self.delta != delta:
                self.delta = delta
                if verbose:
                    tools.warn_(
                        f'Set delta={delta} to make cliprange={cliprange} (adjustdelta_type={adjusttype.name}, dim={dim})')
        # Obtain result
        sigma0 = np.exp(logstd0)
        z = (a-mu0)/ sigma0
        a_1d = np.linalg.norm(z, axis=1) / np.sqrt(dim)
        ratio_min, ratio_max, min_mu_logstd_1d, max_mu_logstd_1d = \
            self.opt1D(action=a_1d, delta=delta / dim, )
        ratio_min = np.power(ratio_min, dim)
        ratio_max = np.power(ratio_max, dim)
        if require_sol:
            assert min_mu_logstd_1d is not None and max_mu_logstd_1d is not None, 'TODO:tabular version does not contain these two variables min_mu_logstd_1d max_mu_logstd_1d'
            sol = self.obtain_sol_from_sol1d( mu0, sigma0, a, z, a_1d, min_mu_logstd_1d, max_mu_logstd_1d, ratio_min, ratio_max, dim=dim )
        else:
            sol = None


        # if clip_clipratio is not None:
        #     assert clip_clipratio >= 1
        #     clip_clipratio_new = ratio_max.min() - 1 + clip_clipratio
        #     print(f'Clip clipratio to {clip_clipratio_new}')
        #     ratio_max = np.minimum(ratio_max, clip_clipratio_new)
        #     ratio_min = np.maximum(ratio_min, 1. / clip_clipratio_new)

        return DotMap(
            ratio=DotMap(max=ratio_max, min=ratio_min),
            sol = sol,
            delta=delta,
        )

    def obtain_sol_from_sol1d(self, mu0_s, sigma0_s, a_s, z_s, a_1d_s, min_mu_logstd_1d_s, max_mu_logstd_1d_s, ratio_min, ratio_max, dim):
        sol = DotMap()
        if dim == 1:
            if a_1d_s.ndim == 1:
                a_1d_s = np.expand_dims( a_1d_s, axis=1  )
            for key, mu_logstd_eq in zip( ['min','max'], [min_mu_logstd_1d_s, max_mu_logstd_1d_s] ):
                # mu, logstd = tools_.mu_logstd_cat2tuple(min_mu_logstd_1d_s)
                R = a_s/np.abs(a_s)
                mu_eq, logstd_eq = tools_.mu_logstd_cat2tuple(mu_logstd_eq)
                mu = sigma0_s * mu_eq * R + mu0_s
                logstd = np.log(sigma0_s) + logstd_eq
                sol[key] = DotMap(mu=mu, logstd=logstd )
                # --- DEBUG
                # p = tools_.p(a_s, mu_logstd=(mu,logstd))
                # p0 = tools_.p(a_s, mu_logstd=(mu0_s, np.log( sigma0_s)))
                # p_eq = tools_.p(a_1d_s, mu_logstd=(mu_eq,logstd_eq))
                # p0_eq = tools_.p(a_1d_s, mu_logstd=( np.zeros_like(mu_eq), np.zeros_like(logstd_eq)) )
                # print(p / p0, p_eq / p0_eq)
                # --- END DEBUG
            sol.min.ratio = ratio_min
            sol.max.ratio = ratio_max
            # print('ratio_min', ratio_min, 'ratio_max', ratio_max)
            # sol_pre = sol # TOD: tmp
            return sol


        for key in ['min','max']:
            sol[key] = DotMap( mu=[], logstd=[], SIGMA=[], mu_eq=[], SIGMA_EQ=[], logstd_eq=[], ratio_debug=[] )
        sol.min.ratio = ratio_min
        sol.max.ratio = ratio_max
        # sol.a_eq = []

        for mu0, sigma0,a, z, a_1d, min_mu_logstd_1d, max_mu_logstd_1d in zip(mu0_s, sigma0_s,a_s, z_s, a_1d_s, min_mu_logstd_1d_s, max_mu_logstd_1d_s):
            #--- Obtain Rotation Matrix R. Using LiWeida's method.
            a_eq = np.ones_like(mu0) * a_1d
            # sol.a_eq.append(a_eq)
            I = np.eye(dim)
            if (z==0).all():
                R= I
            else:
                if dim == 1:
                    R = I * a / np.abs(a)
                else:
                    x1 = a_eq/np.linalg.norm(a_eq)
                    x2 = z / np.linalg.norm(z)
                    x2_t = x2 - dot(x1, x2) * x1
                    x2_hat = x2_t / np.linalg.norm(x2_t)
                    costheta = np.dot(x2, x1)
                    if  abs(abs(costheta)-1) > 1e-8:
                        sintheta = np.dot(x2, x2_hat)
                        R0 = np.array([[costheta, -sintheta], [sintheta, costheta]])
                        X = np.stack((x1, x2_hat), axis=1)
                        R = X @ R0 @ X.transpose() + (I - X @ X.transpose())
                    else:
                        R0 = costheta
                        X = np.expand_dims(x1, axis=1)
                        R = R0 * X @ X.transpose() + (I - X @ X.transpose())
            logstd0 = np.log(sigma0)
            SIGMA0 = np.diag(sigma0)

            #--- DEBUG
            # print('----')
            # print(R@R.transpose(),'\n',R.transpose()@R)
            # print( a_1d, a_eq,'\n' , R.transpose() @ np.linalg.inv(SIGMA0) @ (a-mu0))
            # exit()
            #--- End DEBUG

            def get_sol( mu_logstd_1d, name_pro ):

                target = sol[name_pro]
                mu_eq = mu_logstd_1d[0] * np.ones_like(z)
                mu = SIGMA0 @ R @ mu_eq + mu0

                # 注意：这里的sigma都是标准差
                logstd_eq = mu_logstd_1d[1] * np.ones_like(z)
                SIGMA_EQ = np.diag( np.exp( logstd_eq ) )
                SIGMA = SIGMA0 @ R @ SIGMA_EQ #TODO: check一下，这个结果应该和等价问题的目标函数值是一样的
                # print('---')
                # print(SIGMA@SIGMA.transpose())
                # print( SIGMA.transpose()@SIGMA)
                logstd = np.log( np.sqrt(np.diag(SIGMA@SIGMA.transpose())) ) #TODO: 暂时是直接取对角线的值

                #--- DEBUG
                # - partial 1
                # v_eq = (mu_eq-a_eq) @ np.linalg.inv(SIGMA_EQ @ SIGMA_EQ.transpose()) @ (mu_eq-a_eq)
                # v = (mu-a) @ np.linalg.inv( SIGMA @ SIGMA.transpose() ) @ (mu-a)
                # print(SIGMA0)
                # print( R.transpose()@ np.linalg.inv( SIGMA0 ) @(mu0-a) )
                # print(v_eq, v)

                # - partial 2
                p = scipy.stats.multivariate_normal.pdf(a, mean=mu, cov=SIGMA.transpose()@SIGMA )
                p0 = scipy.stats.multivariate_normal.pdf(a, mean=mu0, cov=SIGMA0@SIGMA0.transpose() )

                # - p
                # p = tools_.p(a, (mu, logstd) )
                # p0 = tools_.p(a, (mu0,logstd0) )
                #
                p_eq = tools_.p(a_eq, mu_logstd=(mu_eq, logstd_eq))
                p0_eq = tools_.p( a_eq, mu_logstd=(np.zeros_like(mu_eq),np.zeros_like(logstd_eq)) )
                print( p/p0, p_eq/p0_eq )
                #--- END DEBUG

                target.mu.append(mu)
                target.logstd.append(logstd)

            get_sol( min_mu_logstd_1d, 'min' )
            get_sol( max_mu_logstd_1d, 'max' )
        for proj in ['min','max']:
            for item in ['mu','logstd']:
                sol[proj][item] = np.stack(sol[proj][item], axis=0)
                # print(f'{item}:', sol[proj][item] )
        # print(sol_pre, '\n', sol)
        # save_vars('t/t.pkl',sol_pre,sol)
        return sol


def tes_cliprange2delta(  ):
    kl2clip = KL2Clip(opt1Dkind='nn')
    kl2clip.tes_atzero()
    # mu0_s = np.array([[0]])
    # logstd0_s = np.array([[0]])
    # a_s = np.array([[1],[-1]])
    # result = kl2clip(mu0_logstd0=(mu0_s, logstd0_s), a=a_s, cliprange=0.2)



if __name__ == '__main__':

    tes_cliprange2delta()
    exit()


    import scipy.stats
    import baselines.ppo2_AdaClip.tools_.tools_ as tools_

    mu0_s = np.array([[0]])
    logstd0_s = np.array([[0]])
    a_s = np.array([[1],[-1]])
    # mu0_s = np.array( [[0,1],[2,1],[3,1]] )
    # logstd0_s = np.array( [[1,0],[2,0],[1,0]] )
    # a_s = np.array([[1, 2], [2, 2], [3, 1]])

    # mu0_s = np.array([[0, 1,0.], [2, 1,0]])
    # logstd0_s = np.array([[1, 0,0.], [2, 0,1.]])
    # a_s = np.array([[1, 2,0.], [2, 2,0]])

    # TODO: test
    # mu0_s = np.array([[0,0], [0,0]])
    # logstd0_s = np.array([[0,0], [0,0]])
    # a_s = np.array([[1,1],[-1,-1]])

    # mu0_s = np.array([[0], [0]])
    # logstd0_s = np.array([[0], [0]])
    # a_s = np.array([[-3], [3]])


    kl2clip = KL2Clip(opt1Dkind='tabular')
    result = kl2clip( mu0_logstd0=(mu0_s, logstd0_s), a=a_s, delta=0.1 )
    print(result)
    # save_vars('t/kl2clip_reduce.pkl',mu0_s, logstd0_s, a_s,result)
    exit()
    #
    result = tools.load_vars('a.pkl')
    target  = result.sol.min
    for mu0, logstd0, a, mu, logstd, ratio in \
            zip(mu0_s, logstd0_s, a_s, target.mu, target.logstd, target.ratio ):

        p = tools_.p(a, (mu, logstd))
        p0 = tools_.p(a, (mu0,logstd0) )
        print( p/p0, ratio )
    # print(result)
    exit()



from scipy.optimize import fsolve
import numpy as np
# from scipy import log, exp
from math import log, exp
import plt_tools
import matplotlib.pyplot as plt
from dotmap import DotMap
# TODO:注意!!!!!!!!!：好像必须要处理成float32很多地方结果才会没问题
def f_setting(pa, delta):
    def f(m):
        m = float(m)
        if (1+m*pa) == 0:
            return None
        part_1 = (m**(1-pa))
        part_2 = ((1/pa+m)**pa)
        part_3 = ((1-pa)/m + pa**2/(1+m*pa))
        part_1_2 = part_1 * part_2
        # return [part_1_2, part_3]
        if isinstance(part_1_2, complex):
            if abs(part_1_2.imag) > 1e-10:
                return None
            part_1_2 = part_1_2.real
        part_1_2_3 = part_1_2 * part_3
        if part_1_2_3 >0:
            return log(  part_1_2_3  )- delta
        return None
    return f

def opt_entity(pa,delta,type='max', sol_ini=None):
    pa, delta = (float(item) for item in (pa, delta))
    f = f_setting(pa, delta)
    if type=='min':
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
    m = fsolve(f,m_ini, full_output=0)

    # lam  = exp(log(abs((m ** p0) * ((1 / p1 + m) ** p1))) - delta)
    # if m < 0:
    #     lam = -lam
    if isinstance(m, np.ndarray) and m.ndim >=1:
        m = m[0]
    m = float(m)
    lam = m**(1-pa) *  (1/pa + m)**pa / exp(delta)
    if isinstance(lam, complex):
        lam = lam.real
    qi_sum = lam * (1 - pa) / m
    qa = lam * pa / (1 / pa + m)
    ratio = qa / pa
    # print(f'type:{type},ratio:{ratio},qa:{qa},kl_constraint:{f(m)},m:{m},lam:{lam}')
    return ratio, m

def opt(pas, delta):
    sol_pre = None
    ratio_maxs = []
    for pa in pas:
        ratio, sol_pre = opt_entity(pa, delta, 'max', sol_pre)
        ratio_maxs.append(ratio)

    sol_pre = None
    ratio_mins = []
    for pa in pas:
        ratio, sol_pre = opt_entity(pa, delta, 'min', sol_pre)
        ratio_mins.append(ratio)
    ratio_maxs = np.array(ratio_maxs)
    ratio_mins = np.array(ratio_mins)
    return DotMap(
        ratio=DotMap(max=ratio_maxs, min=ratio_mins)
    )

def tes_kl2clip_discrete():
    from baselines.ppo2_AdaClip.KL2Clip_discrete.KL2Clip_discrete import KL2Clip
    delta = 0.02
    pas = np.arange(0.01, 1., 0.001)
    kl2clip = KL2Clip(dim=1, opt1Dkind='tabular')
    ress = kl2clip(
        mu0_logsigma0_cat=None, a=None, pas=pas,
        delta=delta,)
    result = opt(pas, delta)
    plt.plot(pas, ress.ratio.max, 'red')
    plt.plot(pas, ress.ratio.min, 'red')
    plt.plot(pas, result.ratio.max, 'blue')
    plt.plot(pas, result.ratio.min, 'blue')
    plt.show()
    plt.pause(1e10)

if __name__ == '__main__':
    tes_kl2clip_discrete()

    delta = 0.01
    pas = np.arange(0.01,0.96,0.001)
    result = opt(pas, delta)
    plt.plot( pas, result.ratio.max, 'blue' )
    plt.plot( pas, result.ratio.min, 'green' )
    plt_tools.set_postion()
    plt.show()
    exit()

def plot():
    pa = 0.9
    delta = 0.01
    f = f_setting(pa, delta)
    ms = np.arange(-10, 3, 0.1)
    fs = np.array([f(m) for m in ms])
    plt.plot( ms, fs, color='blue' )
    plt.scatter( ms[fs==None], np.zeros_like(ms[fs==None]), color='red' )
    plt.ylim([-0.1,0.2])
    plt_tools.set_postion()
    plt.show()
    exit()


def check_positive():
    ms = np.arange(-10, 3, 0.1)
    fs = np.array([f(m) for m in ms])
    fs = (fs>0).transpose()
    # im = np.zeros_like( fs )
    plt.imshow(fs, cmap='gray')
    plt_tools.set_postion()
    plt.show()
    exit()


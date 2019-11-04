from scipy.optimize import fsolve
import numpy as np
# from scipy import log, exp
from math import log, exp
import plt_tools
import matplotlib.pyplot as plt
from dotmap import DotMap
# TODO:注意!!!!!!!!!：好像必须要处理成float32很多地方结果才会没问题
def f_setting(pa, delta):
    def f(r):
        if r < 0 \
            or ((1-pa) / (1 - pa * r))<0:
            return None
        return (1-pa)*log((1-pa) / (1 - pa * r)) - pa * log(r) - delta
    return f

def opt_entity(pa,delta,type='max', sol_ini=None):
    pa, delta = (float(item) for item in (pa, delta))
    f = f_setting(pa, delta)
    if type=='min':
        if sol_ini is not None:
            r_ini = sol_ini
        else:
            r_ini = 1./exp(0.011*(1./pa))
    else:
        if sol_ini is not None:
            r_ini = sol_ini
        else:
            r_ini = exp(0.011*(1./pa))
            while f(r_ini) is None:
                r_ini -= (r_ini-1)*0.9
    r = fsolve(f,r_ini, full_output=0)

    if isinstance(r, np.ndarray) and r.ndim >=1:
        r = r[0]
    return r

def opt(pas, delta):
    ratio_pre = None
    ratio_maxs = []
    for pa in pas:
        ratio_pre = ratio = opt_entity(pa, delta, 'max', ratio_pre)
        ratio_maxs.append(ratio)

    ratio_pre = None
    ratio_mins = []
    for pa in pas:
        ratio_pre = ratio = opt_entity(pa, delta, 'min', ratio_pre)
        ratio_mins.append(ratio)
    ratio_maxs  = np.array(ratio_maxs)
    ratio_mins = np.array(ratio_mins)
    return DotMap(
        ratio=DotMap(max=ratio_maxs, min=ratio_mins)
    )

if __name__ == '__main__':
    delta = 0.01
    pas = np.arange(0.01,1.,0.001)
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


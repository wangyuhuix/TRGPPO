import numpy as np
from baselines import logger
def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)

    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            logger.log(f'ConjugateGraident: Achieve tolernet precision. iters:{i}, Precision:{rdotr}')
            break
    else:
        logger.log(f'ConjugateGraident: Iters used up. iters:{i}, precision: {rdotr}')
    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x


def tes_cg():
    dim = 4
    A =  np.identity(dim) + 0.05 *np.random.normal( size=(dim,dim) )
    A = A @ A.transpose()
    def f_Ax(x):
        return A@x
    g = np.random.normal( size=(dim) )
    b = np.random.normal( size=(dim) )
    delta = 0.5
    delta0 = 0.1
    s = cg( f_Ax, g, cg_iters=100000, verbose=True, residual_tol=1e-20 )
    m = cg( f_Ax, b, cg_iters=100000, verbose=True, residual_tol=1e-20 )
    lam = np.sqrt( s.dot( f_Ax(s) )/ ( 2*( delta-delta0 ) + m.dot(f_Ax(m)) )  )

    x_new = 1./lam * s
    print( delta0+ 1./2 * x_new.dot( f_Ax(x_new) ) - 1./2*m.dot( f_Ax(m)), delta )

    x = x_new - m

    print( delta0 + 1./2 * x.dot( f_Ax(x) ) + b.dot(x) , delta )

    # print(  )



if __name__ == '__main__':
    tes_cg()
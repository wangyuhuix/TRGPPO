from math import log, exp
pa = 0.9
pi = 1 - pa

delta = 0.01
m = 1.9864
m = -2.8073
# m = 1.98640055781677
# print( p0*log(m) + (1-p0) * log( 1/(1-p0)+m ) - delta + log( p0/m + (1-p0)**2/(1+m*(1-p0)) ) )

# p0= '(1-pa)'
# p1= 'pa'
# print( f'log((m**{p0})*((1/{p1}+m)**{p1})*({p0}/m + {p1}**2/(1+m*{p1})))- {delta}' )
# exit()
# print(log((m ** p0) * ((1 / 1 - p0 + m) ** 1 - p0) * (p0 / m + (1 - p0) ** 2 / (1 + m * (1 - p0)))) - delta)

# lam  = exp(log(abs((m ** pi) * ((1 / pa + m) ** pa))) - delta)
# if m < 0:
#     lam = -lam
lam = m**(1-pa) *  (1/pa + m)**pa / exp(delta)
if isinstance(lam, complex):
    lam = lam.real
print(f'm:{m},lam:{lam}')

x_i_sum = lam * pi / m
x_j = lam * pa / (1 / pa + m)
print( x_i_sum, x_j )
print(pi * log(pi / x_i_sum) + pa * log(pa / x_j))
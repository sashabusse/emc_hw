import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


eps0 = 8.84e-12
mu0 = 1.125e-6
# mu as argument is always relative


def f2lamb(f, mu=1., eps=1.):
    return 1./f/np.sqrt(eps*eps0*mu*mu0)


def w2lamb(w, mu=1., eps=1.):
    return f2lamb(w/2/np.pi, mu=mu, eps=eps)


def lamb2f(lamb, mu=1., eps=1.):
    return 1./lamb/np.sqrt(eps*eps0*mu*mu0)


def lamb2w(lamb, mu=1., eps=1.):
    return 2*np.pi*lamb2f(lamb, mu=mu, eps=eps)


def ref_index(mu, eps):
    # коэффициент преломления
    return np.sqrt(mu*mu0*eps*eps0)


def wave_num(w, mu=1., eps=1.):
    # волновое число
    return w*np.sqrt(mu*mu0*eps*eps0)


def wave_res(mu=1., eps=1.):
    # волновое сопротивление
    return np.sqrt(mu*mu0/(eps*eps0))


def A2db(A):
    return 20*np.log10(A)


def P2db(P):
    return 10*np.log10(P)


def db2A(db):
    return 10.**(db/20)


def db2P(db):
    return 10.**(db/10)

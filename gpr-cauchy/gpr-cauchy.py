#!/usr/local/bin/python

import sys
import putil
import numpy as np
from pylab import *
from numpy import exp, log
from elliptical import elliptical
from numpy.linalg import cholesky as chol


def gpr_cauchy(f, param):
    x, y, gamma, Kinv = param
    M = len(x)
    return gpr_cauchy_lik(y[0:M], f[0:M], gamma, Kinv)


def gpr_cauchy_lik(y, f, gamma, Kinv):
    return - np.sum(log(gamma + (y - f)**2 / gamma)) \
           - np.dot(f, np.dot(Kinv, f)) / 2


def kgauss(tau, sigma):
    return lambda x, y: exp(tau) * exp(-(x - y)**2 / exp(sigma))


def kernel_matrix(xx, kernel):
    N = len(xx)
    eta = 1e-6
    return np.array(
        [kernel(xi, xj) for xi in xx for xj in xx]
    ).reshape(N, N) + eta * np.eye(N)


def gpr_mcmc(x, y, iters, xmin, xmax, gamma):
    xx = np.hstack((x, np.linspace(xmin, xmax, 100)))
    M = len(x)
    N = len(xx)
    K = kernel_matrix(xx, kgauss(1, 1))
    Kinv = inv(K[0:M, 0:M])
    S = chol(K)
    f = np.dot(S, randn(N))
    g = np.zeros(len(xx))
    for iter in range(iters):
        f, lik = elliptical(f, S, gpr_cauchy, (x, y, gamma, Kinv))
        g = g + f
        print('\r[iter %2d]' % (iter + 1))
        plot(xx[M:], f[M:])  # color='gray')
    print('')
    plot(x, y, 'bx', markersize=14)
    plot(xx[M:], g[M:]/iters, 'k', linewidth=3)
    # putil.simpleaxis()


def usage():
    print('usage: gpr-cauchy.py data.xyf iters [output]')
    sys.exit(0)


def main():
    xmin = -5
    xmax = 5
    ymin = -7.5
    ymax = 12.5
    gamma = 0.2

    if len(sys.argv) < 3:
        usage()
    else:
        [x, y, f] = np.loadtxt(sys.argv[1]).T
        iters = int(sys.argv[2])

    gpr_mcmc(x, y, iters, xmin, xmax, gamma)
    axis([xmin, xmax, ymin, ymax])

    if len(sys.argv) > 3:
        putil.savefig(sys.argv[3])
    show()


if __name__ == "__main__":
    main()

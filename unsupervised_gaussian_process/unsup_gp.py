#!/usr/local/bin/python

import sys
import numpy as np
from numpy.random import multivariate_normal as mvnrand
import matplotlib.pyplot as plt

xmax = 4
xmin = -4
ymax = 4
N = 5
markers = ['o', '^', 's']
M = len(markers)


def kgauss(params):
    [tau, sigma, eta] = params
    return lambda x, y: \
        np.exp(tau) * np.exp(-(x - y)**2 / np.exp(sigma)) + \
        (np.exp(eta) if (x == y) else 0)


def kernel_matrix(xx, kernel):
    N = len(xx)
    return np.array([kernel(xi, xj) for xi in xx for xj in xx]).reshape(N, N)


def fgp(xx, kernel):
    N = len(xx)
    K = kernel_matrix(xx, kernel)
    return mvnrand(np.zeros(N), K)


def plot_latent():
    n = 0
    tau = np.log(0.5)
    sigma = np.log(1)
    eta = np.log(1e-6)
    ww = (xmax - xmin) * (np.random.randn(N) - 0.5)
    xx = np.hstack((ww, np.linspace(xmin, xmax, 100)))
    for m in range(M):
        ff = fgp(xx, kgauss((tau, sigma, eta))) + 2
        yy = ff[0:N] + 0.2 * np.random.randn(N)
        plt.plot(xx[N:], ff[N:], 'b')
        plt.plot(ww, yy, markers[m], color='black',
                 markersize=12, markerfacecolor='white')
    plt.plot(ww, np.zeros(N), '^k', markersize=20, markeredgewidth=2)
    for x in ww:
        n += 1
        plt.plot([x, x], [0, ymax], color='black',
                 linestyle='dashed', linewidth=1)
        # text(x+0.2, 0.2, r'$x_%d$' % n)
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.show()


def main():
    plot_latent()


if __name__ == "__main__":
    main()

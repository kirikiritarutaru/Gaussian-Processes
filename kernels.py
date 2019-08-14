import sys
import putil
import numpy as np
from pylab import *
from numpy.random import multivariate_normal as mvnrand

# ToDo matplotlibで書き換え
xmax = 5
xmin = -5
ymax = 5
ymin = -5
M = 4
N = 100


def let(val, func):
    return func(val)


def klinear():
    b = randn()
    return lambda x, y: b + x * y


def kexp(sigma):
    return lambda x, y: exp(- abs(x - y) / sigma)


def kgauss(params):
    [tau, sigma] = params
    return lambda x, y: exp(tau) * exp(-(x - y)**2 / exp(sigma))


def kperiodic(params):
    [tau, sigma] = params
    return lambda x, y: exp(tau * cos((x - y) / sigma))


def kmatern3(sigma):
    return lambda x, y: \
        let(abs(x - y), lambda r:
            (1 + sqrt(3) * r / sigma) * exp(- sqrt(3) * r / sigma))


def kmatern5(sigma):
    return lambda x, y: \
        let(abs(x - y), lambda r:
            (1 + sqrt(5) * r / sigma + 5 * r * r / (3 * sigma * sigma))
            * exp(- sqrt(5) * r / sigma))


def kernel_matrix(xx, kernel):
    N = len(xx)
    eta = 1e-6
    return np.array(
        [kernel(xi, xj) for xi in xx for xj in xx]
    ).reshape(N, N) + eta * np.eye(N)


def fgp(xx, kernel):
    N = len(xx)
    K = kernel_matrix(xx, kernel)
    return mvnrand(np.zeros(N), K)


def plot_gaussian():
    xx = np.linspace(xmin, xmax, N)
    for m in range(M):
        plot(xx, fgp(xx, kgauss((1, 1))))


def plot_linear():
    xx = np.linspace(xmin, xmax, N)
    for m in range(M):
        plot(xx, fgp(xx, klinear()))


def plot_exponential():
    xx = np.linspace(xmin, xmax, N)
    for m in range(M):
        plot(xx, fgp(xx, kexp(1)))


def plot_periodic():
    xx = np.linspace(xmin, xmax, N)
    for m in range(M):
        plot(xx, fgp(xx, kperiodic((1, 0.5))))


def plot_matern3():
    xx = np.linspace(xmin, xmax, N)
    for m in range(M):
        plot(xx, fgp(xx, kmatern3(1)))


def plot_matern5():
    xx = np.linspace(xmin, xmax, N)
    for m in range(M):
        plot(xx, fgp(xx, kmatern5(1)))


def usage():
    print('usage: kernels.py kernel [output]')
    sys.exit(0)


def main():
    if len(sys.argv) < 2:
        usage()
    else:
        name = sys.argv[1].lower()

    if name == 'gaussian':
        plot_gaussian()
    elif name == 'linear':
        plot_linear()
    elif name == 'periodic':
        plot_periodic()
    elif name == 'exponential':
        plot_exponential()
    elif name == 'matern3':
        plot_matern3()
    elif name == 'matern5':
        plot_matern5()
    else:
        print('unknown kernel.')
        usage()

    # putil.simpleaxis()
    axis([xmin, xmax, ymin, ymax])

    if len(sys.argv) > 2:
        putil.savefig(sys.argv[2])
    show()


if __name__ == "__main__":
    main()

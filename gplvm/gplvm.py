#!/usr/local/bin/python
#
#    gplvm.py
#    Gaussian Process Latent Variable Model (GPLVM).
#    $Id: gplvm.py,v 1.12 2018/03/30 09:52:38 daichi Exp $
#
import sys
import numpy as np
from numpy.linalg import det, inv
from scipy.optimize import minimize, fmin_l_bfgs_b
# from scg import SCG
from opts import getopts
from printf import eprint


def tr(A, B):
    return (A*B.T).sum()


def printparam(params):
    print(np.array(params[0:5]))


def logdet(A):
    sign, val = np.linalg.slogdet(A)
    return sign * val


def kgauss(params):
    [tau, sigma, eta] = params
    return (lambda x, y:
            np.exp(tau) * np.exp(- np.dot(x-y, x-y) / np.exp(sigma)) +
            (np.exp(eta) if all(x == y) else 0))


def kgaussm(X, params):
    [tau, sigma, eta] = params
    N = len(X)
    x = np.sum(X**2, 1)
    K = np.tile(x, (N, 1)) + np.tile(x, (N, 1)).T - 2 * np.dot(X, X.T)
    return np.exp(tau) * np.exp(- K / np.exp(sigma)) + np.exp(eta) * np.eye(N)


def crossprod(X):
    N = len(X)
    x = np.sum(X**2, 1)
    return np.tile(x, (N, 1)) + np.tile(x, (N, 1)).T - 2 * np.dot(X, X.T)


def kernel_matrix(xx, kernel, params):
    if kernel == kgauss:
        return kgaussm(xx, params)
    else:
        N = len(xx)
        return np.array(
            [kernel(params)(xi, xj) for xi in xx for xj in xx]
        ).reshape(N, N)


def gplvmlik(xx, Y, L, kernel, init):
    H = len(init)
    N, D = Y.shape
    params = xx[0:H]
    X = xx[H:].reshape(N, L)
    K = kernel_matrix(X, kernel, params)
    val = (N * D * np.log(2*pi) + D * np.logdet(K) +
           tr(inv(K), np.dot(Y, Y.T))) / (2 * N)
    eprint('gplvmlik = %.04f' % val)
    return val


def gplvmgrad(xx, Y, L, kernel, init):
    H = len(init)
    N, D = Y.shape
    params = xx[0:H]
    X = xx[H:].reshape(N, L)
    G = np.zeros([N, L])
    K = kernel_matrix(X, kernel, params)
    IK = np.inv(K)
    LK = D * IK - np.dot(IK, Y).dot(Y.T).dot(IK)
    # gradients for hyperparameters
    [tau, sigma, eta] = params
    ga = np.sum(LK * (K - exp(eta) * np.eye(N)))
    gb = np.sum(LK * K * crossprod(X)) / exp(sigma)
    gc = np.sum(diag(LK)) * exp(eta)
    # gradients for latents
    for n in range(N):
        for i in range(L):
            G[n, i] = - 4 * np.sum(LK[n] * (X[n, i] - X[:, i])
                                   * K[n]) / exp(sigma)
    return np.hstack(([ga, gb, gc], G.ravel())) / (2 * N)


def numgrad_h(X, Y, L, kernel, params, eps=1e-6):
    D = len(params)
    ngrad = np.zeros(D)
    for d in range(D):
        lik = gplvmlik(np.hstack((params, X.ravel())),
                       Y, L, kernel, params)
        params[d] += eps
        newlik = gplvmlik(np.hstack((params, X.ravel())),
                          Y, L, kernel, params)
        params[d] -= eps
        ngrad[d] = (newlik - lik) / eps
    return ngrad


def numgrad_x(X, Y, L, kernel, params, eps=1e-6):
    N, D = X.shape
    ngrad = np.zeros(D)
    for d in range(D):
        lik = gplvmlik(np.hstack((params, X.ravel())),
                       Y, L, kernel, params)
        X[0][d] += eps
        newlik = gplvmlik(np.hstack((params, X.ravel())),
                          Y, L, kernel, params)
        X[0][d] -= eps
        ngrad[d] = (newlik - lik) / eps
    return ngrad


def gplvm(Y, L, kernel, optimizer):
    # normalize data
    Y = (Y-np.mean(Y, 0)) / np.sqrt(var(Y, 0))
    # initialize X
    U, S, V = np.svd(Y)
    X = U[:, :L] / 10
    N = len(Y)
    # kernel hyperparameter
    init = [np.log(1), np.log(0.1), np.log(1e-2)]
    # optimize log likelihood
    print('optimizing: optimizer = %s' % optimizer)
    if optimizer in optimizers:
        x = optimizers[optimizer](X, Y, L, kernel, init)
        return x.reshape(N, L)
    else:
        print('unknown optimizer! [%s]' % '|'.join(optimizers.keys()))
        sys.exit(0)


def optimize_lbfgs(X, Y, L, kernel, init):
    H = len(init)
    x, f, d = fmin_l_bfgs_b(
        gplvmlik,
        np.hstack((init, X.ravel())),
        fprime=gplvmgrad,
        args=[Y, L, kernel, init],
        iprint=0,
        maxiter=1000
    )
    return x[H:]


def optimize_bfgs(X, Y, L, kernel, init):
    H = len(init)
    res = minimize(
        gplvmlik, np.hstack((init, X.ravel())),
        args=(Y, L, kernel, init),
        jac=gplvmgrad,
        method='BFGS',  # callback = printparam,
        options={'gtol': 1e-4, 'disp': True}
    )
    print(res.message)
    print('init           =', np.array(init))
    print('hyperparameter =', res.x[0:H])
    return res.x[H:]


def optimize_scg(X, Y, L, kernel, init):
    H = len(init)
    x, flog, feval, status = SCG(
        gplvmlik,
        gplvmgrad,
        np.hstack((init, X.ravel())),
        optargs=[Y, L, kernel, init]
    )
    print(status)
    print('init           =', exp(np.array(init)))
    print('hyperparameter =', exp(x[0:H]))
    return x[H:]


optimizers = {
    'scg': optimize_scg,
    'bfgs': optimize_bfgs,
    'l-bfgs': optimize_lbfgs
}


def plot_latents(X, labels):
    L = X.shape[1]
    colors = ['red', 'blue', 'green', 'cyan']
    if len(labels) > 0:
        for c in range(max(labels)):
            index = ((labels - 1) == c)
            scatter(X[index, 0], X[index, 1], 50, color=colors[c])
    else:
        scatter(X[:, 0], X[:, 1])


def usage():
    print('usage: gplvm.py OPTIONS train [output]')
    print('OPTIONS')
    print('-o optimizer  {scg|bfgs|l-bfgs} (default scg)')
    print('-L latents    latent number of dimensions (default 2)')
    print('-l labels     labels of data to plot')
    print('-h            this help')
    print('$Id: gplvm.py,v 1.12 2018/03/30 09:52:38 daichi Exp $')
    sys.exit(0)


def main():
    options, args = getopts(
        [
            "L|latents=",
            "o|optimizer=",
            "l|labels=",
            "h|help"
        ]
    )
    optimizer = 'bfgs'
    kernel = kgauss
    labels = []
    L = 2

    if len(args) == 0:
        usage()
    else:
        data = np.loadtxt(args[0])
        if 'L' in options:
            L = int(options['L'])
        if 'o' in options:
            optimizer = options['o']
        if 'l' in options:
            labels = np.loadtxt(options['l'], dtype=int)

    latents = gplvm(data, L, kernel, optimizer)

    plot_latents(latents, labels)
    plt.show()


if __name__ == "__main__":
    main()

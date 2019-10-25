#!/usr/local/bin/python

import sys
import GPy
import numpy as np
import matplotlib.pyplot as plt
import putil
from numpy import exp, sqrt


def gpr_poisson(data):
    N = len(data)
    xx = np.linspace(1, N, N)
    model = GPy.core.GP(
        X=xx[:, None],
        Y=data[:, None],
        kernel=GPy.kern.RBF(1),
        inference_method=GPy.inference.latent_function_inference.Laplace(),
        likelihood=GPy.likelihoods.Poisson()
    )
    model.optimize()
    mu, var = model._raw_predict(xx[:, None])
    plt.plot(xx, np.exp(mu))
    plt.fill_between(xx, exp(mu[:, 0] + 3*sqrt(var[:, 0])),
                     exp(mu[:, 0] - 3*sqrt(var[:, 0])),
                     color='#ccccff')
    plt.plot(xx, data, 'xb', markersize=8)
    # plt.plot (xx, data, 'xk', markersize=8)
    # putil.simpleaxis()
    # putil.aspect_ratio(1.3)

    # model.plot ()


def main():
    data = np.loadtxt(sys.argv[1], dtype=int)
    gpr_poisson(data)
    if len(sys.argv) > 2:
        putil.savefig(sys.argv[2])
    plt.show()


if __name__ == "__main__":
    main()

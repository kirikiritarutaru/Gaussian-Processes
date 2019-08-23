import numpy as np
import matplotlib.pyplot as plt


def mean_function(x):
    return np.zeros_like(x)


def kgauss(x, y, s):
    return np.exp(-(x - y) ** 2 / s ** 2)


def klinear(x, y):
    b = np.random.randn()
    return x * y + b


def kexp(x, y, sigma):
    return np.exp(- np.abs(x - y) / sigma)


def kperiodic(x, y, tau, sigma):
    return np.exp(tau * np.cos((x-y) / sigma))


def kmatern3(x, y, sigma):
    r = np.abs(x - y)
    return (1 + np.sqrt(3) * r / sigma) * np.exp(- np.sqrt(3) * r / sigma)


def kmatern5(x, y, sigma):
    r = np.abs(x - y)
    return (1 + np.sqrt(5) * r / sigma + 5 * r * r / (3 * sigma * sigma)) * \
        np.exp(-np.sqrt(3)*r/sigma)


xx = np.linspace(-5, 5, 100)
x, y = np.meshgrid(xx, xx)

# カーネル行列
mean = mean_function(xx)
gram_matrix = kgauss(x, y, 1.0)
linear_matrix = klinear(x, y)
exp_matrix = kexp(x, y, 1.0)
periodic_matrix = kperiodic(x, y, 1.0, 0.5)
matern3_matrix = kmatern3(x, y, 1.0)
matern5_matrix = kmatern5(x, y, 1.0)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

for k in range(5):
    gauss = np.random.multivariate_normal(mean, gram_matrix)
    linear = np.random.multivariate_normal(mean, linear_matrix)
    exp = np.random.multivariate_normal(mean, exp_matrix)
    periodic = np.random.multivariate_normal(mean, periodic_matrix)
    matern3 = np.random.multivariate_normal(mean, matern3_matrix)
    matern5 = np.random.multivariate_normal(mean, matern5_matrix)

    ax[0][0].plot(xx, gauss)
    ax[0][0].set_title('Gaussian')
    ax[0][0].grid(True)

    ax[0][1].plot(xx, linear)
    ax[0][1].set_title("Linear")
    ax[0][1].grid(True)

    ax[0][2].plot(xx, exp)
    ax[0][2].set_title("Exp")
    ax[0][2].grid(True)

    ax[1][0].plot(xx, periodic)
    ax[1][0].set_title("Periodic")
    ax[1][0].grid(True)

    ax[1][1].plot(xx, matern3)
    ax[1][1].set_title("Matern3")
    ax[1][1].grid(True)

    ax[1][2].plot(xx, matern5)
    ax[1][2].set_title("Matern5")
    ax[1][2].grid(True)

plt.show()

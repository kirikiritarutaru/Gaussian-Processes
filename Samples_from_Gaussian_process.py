import numpy as np
import matplotlib.pyplot as plt


def mean_function(x):
    return np.zeros_like(x)


def covariance_function(x1, x2, s):
    return np.exp(-(x1 - x2) ** 2 / s ** 2)


x = np.linspace(-10, 10, 100)
x1, x2 = np.meshgrid(x, x)
sigma = 1.0

mean = mean_function(x)
gram_matrix = covariance_function(x1, x2, sigma)

for k in range(10):
    sample = np.random.multivariate_normal(mean, gram_matrix)
    plt.plot(x, sample, label=f'Sample {k}')

# plt.legend(loc='upper right')
plt.show()

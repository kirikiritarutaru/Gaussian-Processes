import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# プロット用パラメータ
N = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3

# ガウス過程のパラメータ
eta = 0.1
tau = 1
sigma = 1


def kv(x, xtrain):
    return np.array([kgauss(x, xi)for xi in xtrain])


def kgauss(x, y):
    return tau * np.exp(- (x - y)**2 / (2 * sigma * sigma))


def gpr(xx, xtrain, ytrain):
    N = len(xtrain)
    K = np.array(
        [kgauss(xi, xj)for xi in xtrain for xj in xtrain]
    ).reshape(N, N) + eta * np.eye(N)
    Kinv = inv(K)

    ypr = []
    spr = []
    for x in xx:
        s = kgauss(x, x) + eta
        k = kv(x, xtrain)
        ypr.append(k.T.dot(Kinv).dot(ytrain))
        spr.append(s - k.T.dot(Kinv).dot(k))

    return ypr, spr


def main():
    train = np.loadtxt("gpr.dat", dtype=float)
    xtrain = train.T[0]
    ytrain = train.T[1]

    xx = np.linspace(xmin, xmax, N)
    x, y = np.meshgrid(xx, xx)

    x_list = []
    y_list = []
    for xt, yt in zip(xtrain, ytrain):
        x_list.append(xt)
        y_list.append(yt)
        ypr, spr = gpr(xx, x_list, y_list)
        plt.figure(figsize=(10, 5))
        plt.plot(xx, ypr)
        plt.fill_between(
            xx,
            ypr - 2*np.sqrt(spr),
            ypr + 2*np.sqrt(spr),
            color='#ccccff'
        )
        plt.scatter(x_list, y_list, s=300, marker='x')
        plt.pause(0.5)


if __name__ == "__main__":
    main()

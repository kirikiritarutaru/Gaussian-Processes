import sys
import putil
import numpy as np
from pylab import *
from numpy.linalg import inv

xlim = 4
ylim = 2
N = 200


def phi(x):
    return [1, x, x*x, np.sin(x), np.cos(x)]


def lm(xx, w):
    return [np.dot(w, phi(x)) for x in xx]


def add_xy(x=0.0, y=0.0):
    ax = gca().axes
    if x == 0.0 or y == 0.0:
        x = ax.get_xlim()[1] + 0.1
        y = ax.get_ylim()[1] + 0.1
    ax.text(x+0.2, -ylim, r'$x$', va='center', size=18)
    ax.text(-xlim, y+0.1, r'$y$', ha='center', size=18)


def usage():
    print('usage: lm.py data [output]')
    sys.exit(0)


def main():
    data = np.loadtxt(sys.argv[1])

    X = np.array([phi(x) for x in data.T[0]])
    y = data.T[1]
    w = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)
    print(w)

    xx = np.linspace(-xlim, xlim, N)
    yy = lm(xx, w)

    plot(xx, yy)
    plot(data.T[0], data.T[1], 'xk', markersize=12)
    axis([-xlim, xlim, -ylim, ylim])
    yticks(range(-ylim, ylim+1))
    add_xy()

    if len(sys.argv) > 2:
        putil.savefig(sys.argv[2])
    show()


if __name__ == "__main__":
    main()

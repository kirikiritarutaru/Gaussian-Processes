import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


xlim = 4
ylim = 2
N = 200
reg_param = 0.1


# 特徴ベクトル
def phi(x):
    return [1, x, x*x, np.sin(x), np.cos(x)]


# 線形モデル
def linear_model(data, weight):
    # 行列の内積
    return[np.dot(weight, phi(x))for x in data]


# 入力： 線形モデル回帰とリッジ回帰を切り替えフラグ
def main(lm_or_rm):
    data = np.loadtxt("nonlinear.dat")

    # X:入力データの特徴ベクトル
    X = np.array([phi(x) for x in data.T[0]])
    # y:出力
    y = data.T[1]

    if(lm_or_rm):
        # 線形モデルの重みを計算p.31の式(1.67)
        weight = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)
    else:
        # リッジ回帰の場合
        weight = np.dot(
            np.dot(inv(np.dot(X.T, X) + reg_param**np.eye(X.shape[1])), X.T),
            y
        )

    print('feature vector:[1, x, x*x, np.sin(x), np.cos(x)]')
    print('weight:', weight)

    # 線形回帰モデルのプロット
    xx = np.linspace(-xlim, xlim, N)
    yy = linear_model(xx, weight)
    plt.plot(xx, yy)

    # データの散布図
    plt.scatter(data.T[0], data.T[1], marker='x')

    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main(True)

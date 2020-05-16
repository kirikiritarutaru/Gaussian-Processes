import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import TweedieRegressor

sns.set(style='darkgrid')

df = sns.load_dataset('tips')

show_data_plot = False
if show_data_plot:
    sns.jointplot(
        x='total_bill',
        y='tip',
        data=df,
        kind='scatter',
        xlim=(0, 60),
        ylim=(0, 12),
        color='b',
        height=7,
    )
    plt.show()


def sk_poisson_regression(X_train, X_test, y_train, y_test):
    glm = PoissonRegressor(
        alpha=0,
        fit_intercept=False,
        max_iter=300
    )
    glm.fit(X_train, y_train)
    print('score: ', glm.score(X_test, y_test))

    y_hat = glm.predict(X)

    fig = plt.figure(figsize=(6.0, 6.0))
    plt.plot(X, y, 'o')
    plt.plot(X, y_hat, '*', color='r')
    plt.xlabel('x (total_bill)')
    plt.ylabel('y (tips)')
    plt.xlim(0, 60)
    plt.ylim(0, 12)
    plt.show()


def sk_tweedie_regression(
        X_train,
        X_test,
        y_train,
        y_test,
        set_model='linear'
):
    if set_model == 'Poisson':
        reg = TweedieRegressor(
            alpha=0,
            power=1,  # Poisson distribution
            link='log',
            fit_intercept=False,
            max_iter=300
        )
    elif set_model == 'linear':
        reg = TweedieRegressor(
            alpha=0,
            power=0,  # Normal distribution
            link='identity',
            fit_intercept=False,
            max_iter=300
        )
    else:
        print('Set the correct name.')
        return

    reg.fit(X_train, y_train)
    print('score: ', reg.score(X_test, y_test))

    y_hat = reg.predict(X)

    fig = plt.figure(figsize=(6.0, 6.0))
    plt.plot(X, y, 'o')
    plt.plot(X, y_hat, '*', color='r')
    plt.xlabel('x (total_bill)')
    plt.ylabel('y (tips)')
    plt.xlim(0, 60)
    plt.ylim(0, 12)
    plt.show()


if __name__ == '__main__':

    y = df['tip'].values
    X = df['total_bill'].values

    X = X.reshape(len(X), 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sk_tweedie_regression(X_train, X_test, y_train, y_test, set_model='linear')

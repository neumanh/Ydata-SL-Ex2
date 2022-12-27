# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [
# numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html)
# to be more efficient.

import numpy as np


# Classes
##############

class Ols(object):
    """ An implementation on ordinary least squares"""

    def __init__(self):
        self.w = None

    @staticmethod
    def pad(X):
        X = np.c_[X, np.ones(X.shape[0])]
        return X

    def fit(self, X, Y):
        # remember pad with 1 before fitting
        # Update the weight
        self.w = self._fit(X, Y)

    @staticmethod
    def _fit(X, Y):
        X = Ols.pad(X)
        w_ols = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
        return w_ols

    def predict(self, X):
        # return wx
        pred_y = self._predict(X)
        return pred_y

    def _predict(self, X):
        X = self.pad(X)
        pred_y = np.dot(X, self.w)
        return pred_y

    def score(self, X, Y):
        pred_y = self._predict(X)
        mse = np.mean((pred_y - Y) ** 2)
        return mse


# Write a new class OlsGd which solves the problem using gradinet descent. The class should get as a parameter the
# learning rate and number of iteration. Plot the loss convergance. for each alpha, learning rate plot the MSE with
# respect to number of iterations. What is the effect of learning rate? How would you find number of iteration
# automatically? Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to
# normalize your feature first.
class Normalizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)  # The mean of each column (=feature)
        self.stds = np.std(X, axis=0)  # The STD of each column (=feature)

    def predict(self, X):
        norm_x = (X - self.means) / self.stds
        return norm_x

    def __str__(self):
        return f'Normalizer: means: {self.means} stds: {self.stds}'


class OlsGd(Ols):

    def __init__(self, learning_rate=.05,
                 num_iteration=1000,
                 normalize=True,
                 early_stop=True,
                 verbose=True):

        super(OlsGd, self).__init__()
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.early_stop = early_stop
        self.normalize = normalize
        self.normalizer = Normalizer()
        self.verbose = verbose

    def _fit(self, X, Y, reset=True, track_loss=True):
        # remember to normalize the data before starting
        X0 = X  # Saving the original X
        if self.normalize:
            self.normalizer.fit(X)
            X = self.normalizer.predict(X)
        pad_X = self.pad(X)
        self.w = np.random.randint(100, size=pad_X.shape[1])  # Random initial weights
        w0 = np.random.randint(100, size=pad_X.shape[1])  # Random initial weights
        i = 0
        while not self._should_stop_func(i, w0):
            w0 = self.w  # The previous step w, for early stop
            self._step(pad_X, Y)
            i += 1
            if self.verbose:
                loss = self.score(X0, Y)  # Computing the loss
                print(f'Iteration {i} error: {loss}')
        if self.verbose:
            print(f'Final step: iteration: {i} Error: {self.score(X0, Y)}')
        return self.w

    def _derive(self, X, Y):
        """ Derives the w """
        n = X.shape[0]
        grad_x = (2 / n) * np.dot(X.T, (np.dot(X, self.w) - Y))
        return grad_x

    def _predict(self, X):
        if self.normalize:
            X = self.normalizer.predict(X)
        # pred_y = np.dot(X, self.w)
        pred_y = super()._predict(X)

        return pred_y

    def _step(self, X, Y):
        # use w update for gradient descent
        grad_x = self._derive(X, Y)
        self.w = self.w - np.dot(self.learning_rate, grad_x)

    def _should_stop_func(self, current_step, w0, delta=0.001):
        """Returns true if the function should stop"""
        should_stop = False
        if current_step >= self.num_iteration:
            should_stop = True
        elif self.early_stop and (abs(self.w - w0) <= delta).all():
            should_stop = True
        return should_stop


class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs, self).__init__()
        self.ridge_lambda = ridge_lambda

    def _fit(self, X, Y):
        # Closed form of ridge regression
        X = self.pad(X)
        in_brackets = np.dot(X.T, X) + np.dot(self.ridge_lambda, np.identity(X.shape[1]))
        w_ridge = np.dot(np.dot(np.linalg.pinv(in_brackets), X.T), Y)
        return w_ridge


class RidgeGd(OlsGd):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeGd, self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def _derive(self, X, Y):
        """ Derives the w """
        grad_x = super()._derive(X, Y) + self.ridge_lambda * self.w
        return grad_x


# Testing
##############

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    all_X, all_y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.25)

    regression_methods = [Ols, OlsGd, RidgeLs, RidgeGd]
    # Ols
    print('Ols:')
    model = Ols()
    model.fit(X_train, y_train)
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    print('Home model:\t', model.score(X_test, y_test))
    print('SK model:\t', mean_squared_error(y_test, sk_model.predict(X_test)))
    print('-' * 30)

    # OlsGd
    print('OlsGd:')
    model = OlsGd(verbose=False, learning_rate=0.1)
    model.fit(X_train, y_train)
    print('Home model:\t', model.score(X_test, y_test))
    print('SK model:\t', mean_squared_error(y_test, sk_model.predict(X_test)))
    print('-' * 30)

    # RidgeLs
    print('RidgeLs:')
    model = RidgeLs(ridge_lambda=0.01)
    model.fit(X_train, y_train)
    sk_model = Ridge()
    sk_model.fit(X_train, y_train)
    print('Home model:\t', model.score(X_test, y_test))
    print('SK model:\t', mean_squared_error(y_test, sk_model.predict(X_test)))
    print('-' * 30)

    # RidgeGd
    print('RidgeGd:')
    model = RidgeGd(verbose=False, learning_rate=0.1, ridge_lambda=0.01)
    model.fit(X_train, y_train)
    print('Home model:\t', model.score(X_test, y_test))
    print('-' * 30)

    # Lasso
    print('Lasso:')
    sk_model = Lasso()
    sk_model.fit(X_train, y_train)
    print('SK model:\t', mean_squared_error(y_test, sk_model.predict(X_test)))

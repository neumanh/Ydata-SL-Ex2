# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [
# numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html)
# to be more efficient.

import numpy as np


class Ols(object):
    """ An implementation on ordinary least squares"""

    def __init__(self):
        self.w = None

    @staticmethod
    def pad(X):
        print('X.shape', X.shape)

    def fit(self, X, Y):
        # remeber pad with 1 before fitting
        # Update the weight
        self.w = self._derive(X, Y)
        print('------- Fit --------')
        print('self.w.shape', self.w.shape)
        print('X.shape', X.shape)

    @staticmethod
    def _derive(X, Y):
        w_ols = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), Y)
        print('w_ols.shape:', w_ols.shape)
        print('Y.shape', Y.shape)
        return w_ols

    def _fit(self, X, Y):
        # optional to use this
        pass

    def predict(self, X):
        # return wx
        print('------- Predict --------')
        print('self.w.shape', self.w.shape)
        print('X.shape', X.shape)
        pred_y = np.dot(X, self.w)
        print('pred_y.shape:', pred_y.shape)
        return pred_y

    def _predict(self, X):
        # optional to use this
        pass

    def score(self, X, Y):
        # return MSE
        pass


# TESTING
n, k = 10, 2
# set the dimensions of the design matrix
beta = np.array([1, 1, 10])  # set the true coefficients
x = np.concatenate([np.ones((n, 1)), np.random.randn(n, k)], axis=1)  # generate random x
y = np.matmul(x, beta) + np.random.randn(n)  # generate random y
print('Inputs:')
print('x:', x.shape)
print('y:', y.shape)

ols_obj = Ols()
ols_obj.fit(x, y)


print(ols_obj.predict(x))
print('true y:\n', y)


# Write a new class OlsGd which solves the problem using gradinet descent. The class should get as a parameter the
# learning rate and number of iteration. Plot the loss convergance. for each alpha, learning rate plot the MSE with
# respect to number of iterations. What is the effect of learning rate? How would you find number of iteration
# automatically? Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to
# normalize your feature first.
class Normalizer():  ## For the GD
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        # apply normalization
        pass


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
        # remeber to normalize the data before starting
        pass

    def _predict(self, X):
        # remeber to normalize the data before starting
        pass

    def _step(self, X, Y):
        # use w update for gradient descent
        pass


class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs, self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def _fit(self, X, Y):
        # Closed form of ridge regression
        pass

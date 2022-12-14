# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.

class Ols(object):
  def __init__(self):
    self.w = None
    print('Tamp3')
    
  @staticmethod
  def pad(X):
    pass
  
  def fit(self, X, Y):
    #remeber pad with 1 before fitting
    pass
  
  def _fit(self, X, Y):
    # optional to use this
    pass
  
  def predict(self, X):
    #return wx
    pass
  
  def _predict(self, X):
    # optional to use this
    pass
    
  def score(self, X, Y):
    #return MSE
    pass


# Write a new class OlsGd which solves the problem using gradinet descent. 
# The class should get as a parameter the learning rate and number of iteration. 
# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
# What is the effect of learning rate? 
# How would you find number of iteration automatically? 
# Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your feature first.
class Normalizer():
  def __init__(self):
    pass

  def fit(self, X):
    pass

  def predict(self, X):
    #apply normalization
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
    #remeber to normalize the data before starting
    pass
        
  def _predict(self, X):
    #remeber to normalize the data before starting
    pass
      
  def _step(self, X, Y):
    # use w update for gradient descent
    pass


class RidgeLs(Ols):
  def __init__(self, ridge_lambda, *wargs, **kwargs):
    super(RidgeLs,self).__init__(*wargs, **kwargs)
    self.ridge_lambda = ridge_lambda
    
  def _fit(self, X, Y):
    #Closed form of ridge regression
    pass

"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    #err = None
    err = np.square(np.linalg.norm(np.matmul(X, w) - y)) / len(y)
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  #w = None
  temp = np.linalg.inv(np.matmul(X.T, X))
  w = np.dot(np.matmul(temp, X.T), y)
  
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    #w = None
    inv_x = np.matmul(X.T, X)
    invertible = False
    while not invertible:
      try:
        inv = np.linalg.inv(inv_x)
      except np.linalg.LinAlgError:
        #Not invertible. Skip this one.
        inv_x += 0.1 * np.identity(len(X[0]))
      else:
        invertible = True
    w = np.matmul(np.matmul(inv, X.T), y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    #w = None
    inv = np.linalg.inv(np.matmul(X.T, X) + lambd * np.identity(len(X[0])))
    w = np.matmul(np.matmul(inv, X.T), y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = -19
    bestErr = 1
    for p in range(-19, 20):
      l = 10 ** p
      w = regularized_linear_regression(Xtrain, ytrain, l)

      err = mean_square_error(w, Xval, yval)
      if err < bestErr:
        bestErr = err
        bestlambda = l

    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X: (num_training_samples, D * power)
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################	
    ret_x = X
    for i in range(2, power + 1):
      X = np.square(X)
      ret_x = np.append(ret_x, X, axis = 1)
    
    return ret_x

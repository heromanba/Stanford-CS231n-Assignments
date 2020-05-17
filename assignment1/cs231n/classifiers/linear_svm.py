import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #initialize the gradient as zero
  dW = np.zeros(W.shape)
  
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] = X[i] + dW[:,j]
        dW[:,y[i]] = - X[i] + dW[:,y[i]]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW + 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
    
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  #compute score for each training example
  scores = X.dot(W)
  #find scores of the correct classes
  correct_scores = np.array([scores[i, y[i]] for i in range(num_train)])
  #compute the loss matrix
  loss_matrix = np.maximum(0, scores - correct_scores.reshape(num_train,1) + 1)
  #compute the loss
  data_loss = (np.sum(loss_matrix) - num_train) / num_train
  reg_loss = reg * np.sum(W * W)
  loss = data_loss + reg_loss

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #find positive values of the loss_matrix and set to 1
  c = (loss_matrix > 0) * np.ones(loss_matrix.shape)
  #suppose the number of positive numbers in one row is n
  #then the gradient contributed by the correct class is (-n+1)
  for i in range(num_train):
      c[i, y[i]] = - np.count_nonzero(c[i,:] != 0) + 1
  #compute the contribution of each dimension over the entire training set
  data_dW = X.transpose().dot(c) / num_train
  #compute the gradient
  dW = data_dW + 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

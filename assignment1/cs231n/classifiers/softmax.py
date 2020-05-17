import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  dim = W.shape[0]
  for i in range(num_train):
        #calculate scores
        f_i = X[i].dot(W)
        #numerical stability: find the highest score and subtract
        f_i -= max(f_i)
        #calculate the probability
        p_i = np.exp(f_i) / np.sum(np.exp(f_i))
        #calculate the loss
        loss -= np.log(p_i[y[i]])
        dW += X[i].reshape(dim, 1).dot(p_i.reshape(1, num_class))
        dW[:, y[i]] -= X[i]
  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores_initial = X.dot(W)
  #numerical stability
  scores = scores_initial - np.amax(scores_initial, axis=1).reshape(num_train, 1)
  #calculate the probability
  prob = np.exp(scores) / np.exp(scores).sum(1).reshape(num_train, 1)
  #create a matrix to find the probability of the correct class
  c = np.zeros(prob.shape)
  for i in range(num_train):
        c[i, y[i]] = 1
  #get the probability of the correct class
  d = prob * c
  #get all positive numbers of that matrix and calculate the data loss
  loss = - np.sum(np.log(d[d>0]))
  #calculate the dW and loss with regularization
  dW = X.transpose().dot(prob - c) / num_train + 2 * reg * W
  loss = loss / num_train + reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


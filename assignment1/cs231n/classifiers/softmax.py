import numpy as np
from random import shuffle

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
  N = X.shape[0]
  K = W.shape[1]
  for i in xrange(N):
    f = np.dot(X[i], W) # unnormalized prob for all classes
    f -= np.max(f) # numerical stability
    exp_f = np.exp(f)
    sum_f = np.sum(exp_f)
    p = exp_f[y[i]] / sum_f # normalized prob
    loss += -np.log(p)
    for j in xrange(K):
      p = exp_f[j] / sum_f # normalized prob
      dW[:, j] += (p - (y[i]==j)) * X[i, :]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= N
  dW /= N

  # L2 regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

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
  N = X.shape[0]
  K = W.shape[1]

  f = np.dot(X, W) # N x K
  f -= f.max(axis=1).reshape((N, 1)) # f = scores
  exp_f = np.exp(f)
  sum_f = np.sum(exp_f, axis=1) # N x 1

  p = exp_f / sum_f.reshape((N, 1)) # N x K
  logp_y = np.log(p[np.arange(N), y])
  loss = np.sum(-logp_y)
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)

  p[np.arange(N), y] -= 1
  dW = np.dot(X.T, p)
  dW /= N
  dW += reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


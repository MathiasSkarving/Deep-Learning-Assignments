import random
from cmath import sqrt

import math
import matplotlib.pyplot as plt
import numpy as np

'''
A little help to understand the shapes

X          (500, 2353)     input images
W          (2353, 8)       weights
scores     (500, 8)        matmul  <- the only shape-changing step so far
correct    (500, 1)        one number per image
margins    (500, 8)        one penalty per image per class
.sum(axis=1)   (500,)      one penalty per image   <- axis 1 collapses
.mean()        ()          a single number         <- everything collapses
'''

def svm_loss(W, X, y, reg=0.0) -> tuple[float, np.ndarray]:
  N = X.shape[0]
  rows = np.arange(N)
  scores = X @ W
  correct = scores[rows, y][:, None] # y basically says: Which class is the "CORRECT" one? for each image in rows, then the output is just a list of the correct scores,
  # where the index is the image "id" and the value is the correct score
  margins = np.maximum(0, scores - correct + 1) # This calculates, how much does each class' score lose to the correct class' score, so it outputs 8 (the number of classes) values for each image
  margins[rows, y] = 0  # the j == y_i term must not count. This creates a for loop (for i in rows, take y[i]) y[i] reveals the index that margins should ignore
  # because the loss function should only sum over all classes except the true class

  losses = np.sum(margins, axis=1)
  data_loss = losses.mean()
  reg_loss = reg * np.sum(W * W) # Penalize large weights
  loss = data_loss + reg_loss

  g = (margins > 0).astype(float) # Matrix of 1's and 0's. Tells which way to push
  g[rows, y] = -g.sum(axis=1) # The correct class needs to win against all the other ones, so it gets pushed in the correct direction with a strength that equals the amount of close or winning contestants
  g /= N # The average is taken
  dW = X.T @ g + 2 * reg * W  

  return loss, dW

def softmax_loss(W, X, y, reg=0.0) -> tuple[float, np.ndarray]:
  N = X.shape[0]
  rows = np.arange(N)
  scores = X @ W
  scores_shifted = scores - np.max(scores) # for safety, for not reaching a bit overflow

  exp_scores = np.exp(scores_shifted)
  probs = exp_scores / np.sum(exp_scores)
  correct_logprobs = -np.log(probs[rows, y])

  losses = np.sum(correct_logprobs)
  data_loss = losses.mean()
  reg_loss = reg * np.sum(W * W)
  loss = data_loss + reg_loss

  gradient_scores = probs.copy()
  gradient_scores[rows, y] -= 1 # The correct probabilities are negated by one to push their weights up
  gradient_scores /= N

  dW = X.T @ gradient_scores + (2 * reg * W) # apply the same logic as with SVM, include L2 regulation to penalize larger weights

  return loss, dW







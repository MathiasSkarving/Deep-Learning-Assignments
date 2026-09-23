from enum import Enum
import math
import numpy as np

class Loss_type(Enum):
    SOFT_MAX = 0,
    SVM = 1
    
class LinearClassifier:    
    def __init__(self, input_dim, num_classes, loss_type=Loss_type.SOFT_MAX):
        self.W = np.random.randn(input_dim, num_classes) * 0.01 
        self.loss_type = loss_type

    def train(self, X, y, learning_rate=0.001, reg=0.00001, num_iters=1000, batch_size=200):
        num_train = X.shape[0]
        loss_history = []
        
        for it in range(num_iters):
            # 1. Randomly sample a mini-batch of batch_size indices
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            Y_batch = y[batch_indices]
            
            # 2. Compute loss and gradient according to loss_type
            if self.loss_type == Loss_type.SOFT_MAX:
                loss, gradient = self.softmax_loss(self.W, X_batch, Y_batch, reg)
            elif self.loss_type == Loss_type.SVM:
                loss, gradient = self.svm_loss(self.W, X_batch, Y_batch, reg)    
                
            loss_history.append(loss)
            
            # 3. Update weights using Gradient Descent
            self.W -= learning_rate * gradient
        return loss_history        
    
    def predict(self, X: list[int]):
        scores = X @ self.W
        return np.argmax(scores, axis=1)

    def svm_loss(self, W, X, y, reg=0.0) -> tuple[float, np.ndarray]:
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

    def softmax_loss(self, W, X, y, reg=0.0) -> tuple[float, np.ndarray]:
        N = X.shape[0]
        rows = np.arange(N)
        scores = X @ W
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True) # for safety, for not reaching a bit overflow

        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[rows, y])
        
        data_loss = np.mean(correct_logprobs)
        reg_loss = reg * np.sum(W * W)
        loss = data_loss + reg_loss

        gradient_scores = probs.copy()
        gradient_scores[rows, y] -= 1 # The correct probabilities are negated by one to push their weights up
        gradient_scores /= N

        dW = X.T @ gradient_scores + (2 * reg * W) # apply the same logic as with SVM, include L2 regulation to penalize larger weights

        return loss, dW

"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = 0  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1/(1+np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        NUM_REPORTS = 10
        N, D = X_train.shape
        LOWER_LR = True

        
        # For this assignment, the bias w_0 will be at the begining of the weight vector
        self.w = np.zeros(D+1)
        X_train_extended = np.append(np.ones(N).reshape(N,1), X_train, axis=1)

        for epoch in range(self.epochs):
            # error will be a (D,) vector 
            error = y_train - self.sigmoid(np.dot(X_train_extended, self.w))
            # w_change is a (N,) vector
            w_change = np.sum((X_train_extended.T * error), axis=1)
            if LOWER_LR:
                scaled_lr = self.lr - (self.lr / (self.epochs - epoch))
                self.w += scaled_lr * w_change
            else:
                self.w += self.lr * w_change

            if epoch % (np.ceil(self.epochs / NUM_REPORTS)) == 0:
                print(f"Epoch {epoch}: unweighted max change in weights = {np.max(np.abs(w_change))}")
            
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
            
        return [1 if self.w[0] + np.dot(self.w[1:], X) > self.threshold else 0 for X in X_test]

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
        self.w = np.zeros(X_train.shape[1]+1)
        
        def P1_given_X(X):
            return self.sigmoid(self.w[0] + np.dot(self.w[1:], X))
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        for epoch in range(self.epochs):
            print(f"starting epoch {epoch}")
            sum_term = 0
            for j in range(X_train.shape[0]):
                sum_term += y_train[j] - P1_given_X(X_train[j])
            self.w[0] = self.w[0] + self.lr*sum_term
            
            for i in range(len(self.w)-1):
                sum_term = 0
                for j in range(X_train.shape[0]):
                    sum_term += X_train[j][i] * (y_train[j] - P1_given_X(X_train[j]))
                self.w[i+1] = self.w[i+1] + self.lr*sum_term
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
            
        return [1 if self.w[0] + np.dot(self.w[1:], X) > 0.5 else 0 for X in X_test]

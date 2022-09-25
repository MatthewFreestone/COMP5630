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
        STEP_REPORTS = 5
        
        N, D = X_train.shape
        # For this assignment, the bias w_0 will be at the end of the weight vector
        self.w = np.zeros(D+1)
        # self.w = np.random.normal(0, 0.5, size=(X_train.shape[1]+1))
        X_train_extended = np.append(np.ones(N).reshape(N,1), X_train, axis=1)

        # P1_given_X = lambda X: self.sigmoid(self.w[0] + np.dot(self.w[1:], X))
        # P1_given_X_extended = lambda X: self.sigmoid(np.dot(X, self.w))
        for epoch in range(self.epochs):
            # error will be a (D,) vector 
            error = y_train - self.sigmoid(np.dot(X_train_extended, self.w))
            # possible_w_change = self.lr * np.sum((X_train_extended.T * error), axis=1)
            max_change = 0
            for i in range(D+1):
                # X_train_extended[:,i] gets the ith column from X
                delta = np.dot(X_train_extended[:,i], error)
                self.w[i] = self.w[i] + self.lr * delta
                max_change = max(np.abs(delta), max_change)

            if epoch % (self.epochs // STEP_REPORTS + 1) == 0:
                print(f"Finished Epoch {epoch}, max_change {max_change}")

        # for epoch in range(self.epochs):
        #     print(f"starting epoch {epoch}")
        #     sum_term = 0
        #     for j in range(X_train.shape[0]):
        #         sum_term += y_train[j] - P1_given_X(X_train[j])
        #     self.w[0] = self.w[0] + self.lr*sum_term
            
        #     for i in range(len(self.w)-1):
        #         sum_term = 0
        #         for j in range(X_train.shape[0]):
        #             sum_term += X_train[j][i] * (y_train[j] - P1_given_X(X_train[j]))
        #         self.w[i+1] = self.w[i+1] + self.lr*sum_term
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

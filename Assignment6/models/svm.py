"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = np.array([0]) 
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        assert self.n_class == 2

    def cost(self, X_test, y_test):
        # N = X_test.shape[0]
        distances = 1 - y_test * np.dot(X_test, self.w)
        distances[distances < 0] = 0
        hinge_loss = self.reg_const * np.sum(distances)
        return 0.5 * np.dot(self.w,self.w) + hinge_loss

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        N,D = X_train.shape
        # N,   1     N,      N,
        dist = 1 - y_train * np.dot(X_train, self.w)

        # Matrix N,D    1             N,1                   N,D
        updates = self.reg_const * y_train[:,np.newaxis] * X_train
        # Matrix N,d          Maxtrix N,1                   N,D
        updates = (dist > 0).astype(int)[:,np.newaxis] * updates
        updates = -updates + self.w
        
        # D Vector 
        return np.sum(updates, axis=0)/N

        ## Less vectorized version, about 15x slower
        # N,D = X_train.shape
        # dist = 1 - y_train * np.dot(X_train, self.w)
        # res = np.zeros(D)
        # for i, v in enumerate(dist):
        #     if max(0,v) != 0:
        #         res += self.w - (self.reg_const * y_train[i] * X_train[i])
        #     else:
        #         res += self.w
        # return res/N

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # transform y_train from {0,1} to {-1,1}
        y_train = np.where(y_train == 0, -1, 1)

        # add column for bias to X
        X_train = np.append(np.ones((X_train.shape[0],1)), X_train, axis=1)

        N,D = X_train.shape
        BATCH_SIZE = N

        # w0 is implicitly at the beginning
        self.w = np.zeros(D)
        for _ in range(self.epochs):
            for i in range((N // BATCH_SIZE)):
                X_curr, y_curr = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE],  y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                self.w -= self.alpha * self.calc_gradient(X_curr, y_curr)
        return

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
        X_test = np.append(np.ones((X_test.shape[0],1)), X_test, axis=1)
        N,D = X_test.shape
        assert self.w.shape[0] == D
        pred = np.dot(X_test, self.w)

        return (pred > 0).astype(int)
        # pred = np.array([])
        # for i in range(X_test.shape[0]):
        #     yp = 1 if np.dot(self.w, X_test[i]) > 0 else 0
        #     pred = np.append(pred, yp)
        # return pred
import numpy as np
class Perceptron:
    def __init__(self, d):
        self.w = np.zeros(d+1)
        self.activation_threshhold = 0.5

    def sigmoid(self, z:np.ndarray):
        return 1/(1+np.exp(-z))

    def predict(self, X: np.ndarray, append=False):
        if append:
            X = np.append(X, np.ones(X.shape[0]).reshape(-1,1), axis=1)
        res = self.sigmoid(np.dot(X, self.w))
        return res

    def train(self, X: np.ndarray, Y: np.ndarray, append=False, lr=0.01):
        if append:
            X = np.append(X, np.ones(X.shape[0]).reshape(-1,1), axis=1)
        res = self.predict(X, append=False)
        self.w += (lr * np.dot((Y-res).reshape(1,-1), X)).reshape(-1)
    
    def accuracy(self, X: np.ndarray, Y_actual):
        pred = self.predict(X, append=False)
        res = np.where(pred > self.activation_threshhold, 1, 0) 
        return np.sum(res == Y_actual) / len(Y_actual)


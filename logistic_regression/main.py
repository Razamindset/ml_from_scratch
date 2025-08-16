import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iters):
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}/{self.n_iters}")

            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            # Gradients: See the math on register

            # dL/dw
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))

            # dL/db
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

if __name__ == "__main__":
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = LogisticRegression(lr=0.01)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    def accuracy(y_pred, y_test):
        return np.sum(y_pred==y_test)/len(y_test)

    acc = accuracy(y_pred, y_test)
    print(acc)

import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN():
    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y 

    def _predict(self, x):
        
        # Calc the distances for x to all the other points
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]

        # Sort 
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_labels).most_common()

        return most_common[0][0]

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions



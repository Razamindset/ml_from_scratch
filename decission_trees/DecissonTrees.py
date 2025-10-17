import numpy as np
from collections import Counter

# Compute the gini impurity for a sample 
def gini(y: np.array):
    # Gini= 1 − ∑ ​pi ** 2​
    # pi is the probability of class i being in the node.

    # First count the occurences for each class in this node
    counts = Counter(y)

    impurity = 1

    for label in counts:
        p = counts[label] / len(y) # probability = occurence_count / total_samples
        impurity -= p ** 2

    return impurity

# * Very Nice
# print(gini([0, 0, 1, 1]))  # 0.5
# print(gini([0, 0, 0, 0]))  # 0.0 (pure)

# Calculate the information gain for this node
def information_gain(y, y_left, y_right):
    n = len(y)
    n_right = len(y_right)
    n_left = len(y_left)

    # Formula: IG = gini(parent) - ( nLeft/n_samples * gini(left) + nRight/n_samples * gini(right))
    gain = gini(y) - ( (n_left / n) * gini(y_left) + (n_right / n) * gini(y_right) )
    
    return gain

# # Sample data
# y = ['A', 'A', 'B', 'B', 'B']
# y_left = ['A', 'A']
# y_right = ['B', 'B', 'B']

# # Compute information gain
# gain = information_gain(y, y_left, y_right)
# print(f"Information Gain: {gain:.3f}")

# Split the data according to a specific feature and threshold
def split_data(X, y, feature, threshold):
    # X is 2D arary of n_samples and features
    # y is 1D array of labels
    # feature is the idx of the feature we should use to split at threshold

    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

class Node:
    def __init__(self, feature=None, threshold=None, right=None, left=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.right =right
        self.left = left
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecissionTree:
    def __init__(self):
        pass

    def fit(self):
        pass

    def  predict(X, y):
        pass


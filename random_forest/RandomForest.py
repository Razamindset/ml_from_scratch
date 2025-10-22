from ..decission_trees.DecissonTrees import DecisionTree
import numpy as np 
from sklearn.utils import resample
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=5, max_depth=5, sample_size=None, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.n_features = n_features
        self.trees = []

    def fit(self, X: np.array, y: np.array):
        n_samples, n_total_features = X.shape
        self.sample_size = self.sample_size or n_samples
        self.n_features = self.n_features or n_total_features

        self.trees = []

        for i in range(self.n_trees):

            # Resample the data for the trees. with replacement=True
            # * Pick a random subset of rows
            X_sample, y_sample = resample(X, y, n_samples=self.sample_size, random_state=i)

            # * Pick a random subset of columns
            # Randomly sample feature inedexes.
            feature_idx = np.random.choice(n_total_features, self.n_features, replace=False)

            tree = DecisionTree(max_depth=self.max_depth)
            
            # Now fit the decission tree with newly selected random data 
            tree.fit(X_sample[:, feature_idx], y_sample)

            # Store the tree and the features it used to split the data
            self.trees.append((tree, feature_idx))


    def predict_one(self, x):
        preds = []

        # We need to go over each tree and get its prediction.
        for tree, feature_idxs in self.trees:
            # For prediction the tree can only use the features it eas trained on
            x_subset = x[feature_idxs]
            preds.append(tree.predict_one(x_subset, tree.root))
        
        most_common = Counter(preds).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
        pass
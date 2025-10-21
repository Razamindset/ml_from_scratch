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

class DecisionTree:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        # * The flow is something like this.
        # 1. We start at a node. Inorder to go to the next node we need to find the feature and a threshold
        # that is most suitable. i.e age <= 18 years or Gender

        # 2. We split the data based on all the features and threshold one by one and calc information gain of each split.

        # 3. To calculate the information gain we use an impurity metric like gini impurity.

        # 4. Now we take the best feature and split the data into nodes based on feature.
        
        # 5. if we started from 1 node where the feature X(binary) was mixed. The each child will have 
        # each type of node. left will have say 1 and right will have say 0.
        
        # 6. We will recursively repeat this until we reach a termination condition that is either we have completely
        # purified the data or we have reached the max depth that we can do recursively. 

        # Note: there are many other impurity metrics like entropy which is similar to KL divergance. (See notes for KL divergance)

        # this will be a recursive call
        # The possible terminatopn are two states
        # either max depth reached or all the nodes are same class
        if len(set(y)) == 1 or depth >= self.max_depth:
            # Label the data based on majority split
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)
        

        n_samples, n_features = X.shape
        best_gain = 0
        best_split = None

        # Lets try all features and thresholds to calculate the best one
        for feature in range(n_features):

            # For thresholds we try all possible values for a feature

            # ! This implemnetaion feeels personally overwhelming

            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right =  split_data(X, y, feature, threshold=threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue # Not enough juice

                gain = information_gain(y, y_left, y_right) 

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, X_left, X_right, y_left, y_right)
                
        if best_gain == 0:
            # Can't improve impurity — make leaf
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)
        
        # At this point we have completed the folowing
        # Best threshold and best gain for that threshold
        # Split the data according to the best calculated thresholds.
        # After splitting now we can do recursive calls for the left and right data

        #* Recursion Babbyyyyyyyy
        feature, threshold, X_left, X_right, y_left, y_right = best_split
        left_node = self.fit(X_left, y_left, depth + 1) # this will be called recursively until reaches leaf
        right_node = self.fit(X_right, y_right, depth + 1)

        # The folowing code will run after all the recusive calls are complete
        # We can assume that it is a parent and we can complete our recursion.

        node = Node(feature, threshold, left=left_node, right=right_node)
        if depth == 0:
            self.root = node
            return self.root
        else:
            return node

    def predict_one(self, X, node: Node):
        if node.is_leaf():
            return node.value
        
        if X[node.feature] <= node.threshold:
            return self.predict_one(X, node.left)

        else:
            return self.predict_one(X, node.right)
    
    def predict(self, X):
        # Take each row and call predictions
        assert hasattr(self, "root") , "Error: Tree not fiited yet."
        return np.array([self.predict_one(x, self.root) for x in X])

X = np.array([
    [2.7, 2.5],
    [1.3, 1.5],
    [3.0, 3.5],
    [0.8, 0.6],
])

y = np.array([1, 0, 1, 0])

tree = DecisionTree(max_depth=2)
tree.fit(X, y)

print(tree.predict(np.array([[1.0, 1.0], [3.0, 3.0]])))  # → [0, 1]

### It works OMY.
### I am the best programmer in the world.
### God choose me to this. 
### I am basically blessed.

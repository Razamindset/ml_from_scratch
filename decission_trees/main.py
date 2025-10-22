from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecissonTrees import DecisionTree

# Load data
wine = load_wine()
X, y = wine.data, wine.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train your custom tree
tree = DecisionTree(max_depth=4)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

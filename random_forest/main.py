import numpy as np
from sklearn.utils import resample

# Example dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

X_sample, y_sample = resample(X, y, n_samples=5, random_state=69)

print(X_sample.flatten())

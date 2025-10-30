from SVM import SVM
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from visulize import visualize_svm


#! There is still something sketch here 
#! Changing the random state and the std is diverging the model too much
#! Need to look at this villian later

X, y = datasets.make_blobs(
    n_samples=200, n_features=2, centers=2, cluster_std=0.6, random_state=42
)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title("Linearly Separable Data (cluster_std=0.6)")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=69)

model = SVM(n_iters=3000, learning_rate=0.1, lambda_param=0.001)

model.fit(X_train, y_train)

visualize_svm(model, X_train, y_train)

predictions = model.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

print(f"Accuracy of SVM model is: {accuracy(y_test, predictions)*100} %")


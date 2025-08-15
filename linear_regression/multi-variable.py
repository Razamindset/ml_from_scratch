import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./linear_regression/dataset.csv")


def multi_input_gradient_decent(m, b, points, lr):
    m_grad = np.zeros(len(m))
    b_grad = 0

    n = len(points)

    for i in range(len(points)):
        X = points.iloc[i][:-1].values
        y = points.iloc[i].Score

        # Here X will be a row vector which has one value for each feature
        # M is also a row vector where colums are equal to the number of features
        # the dot prodeuct will do the follwoign calculation y_pred = h(x) = m1x1 + m2x2 + ... + b
        y_pred = np.dot(m, X) + b
        error = (y - y_pred)

        # Calculate the gradients using the formaula for derivative of loss
        # -2/n can be done either inside or outside the loop
        m_grad +=  X * error
        b_grad +=  error

    m_grad =  -(2/n) * m_grad 
    b_grad = -(2/n) * b_grad

    m = m - lr * m_grad
    b = b - lr * b_grad

    return m, b

num_features = df.shape[1] - 1
m = np.zeros(num_features)
b = 0
lr = 0.001
epochs = 100


for i in range(epochs):
    if i % 10 == 0:
        print(f"Epoch: {i}")

    m, b = multi_input_gradient_decent(m, b, df, lr)

print(m,b)

y_pred = df[['Hours_Study', 'Sleep_Hours']].values @ m + b
plt.scatter(df.Score, y_pred, color='blue')
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.plot([df.Score.min(), df.Score.max()],
         [df.Score.min(), df.Score.max()],
         color='red', linestyle='--')
plt.show()

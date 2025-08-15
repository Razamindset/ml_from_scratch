import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./linear_regression/dataset.csv")

# For understanding
def loss_function(m, b, data):
    total_error = 0

    for i in range(len(data)):
        X = data.iloc[i].Hours_Study
        y = data.iloc[i].Score

        # See the math in notes
        total_error += ( y - ( m * X + b ) ) ** 2

    total_error // float(len(data))
    return total_error

def gradient_decent(m_now, b_now, points, lr):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        X = points.iloc[i].Hours_Study
        y = points.iloc[i].Score

        m_gradient += -(2/n) * X * (y - ( m_now * X + b_now))
        b_gradient += -(2/n) * (y - ( m_now * X + b_now))
    
    m = m_now - m_gradient * lr
    b = b_now - b_gradient * lr

    return m, b 

m = 0
b = 0
lr = 0.001
epochs = 100

for i in range(epochs):
    if i % 10 == 0:
        print(f"Epoch: {i}")

    m, b = gradient_decent(m, b, df, lr)


print(m,b)
plt.scatter(df.Hours_Study, df.Score)
plt.plot(list(range(0, 10)), [m * X + b for X in range(0, 10)], color="red")
plt.show()
import matplotlib.pyplot as plt
import numpy as np

def visualize_svm(model, X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7, s=50)

    # Create grid to evaluate model
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = np.dot(xy, model.w) + model.b
    Z = Z.reshape(XX.shape)

    # Decision boundary and margins
    plt.contour(
        XX, YY, Z,
        colors=['k', 'k', 'k'],
        levels=[-1, 0, 1],
        alpha=0.8,
        linestyles=['--', '-', '--']
    )

    plt.title("SVM Decision Boundary and Margins")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
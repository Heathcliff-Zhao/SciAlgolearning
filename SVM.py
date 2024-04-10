import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

class KernelSVM:
    def __init__(self, kernel=lambda x, y, gamma: np.dot(x, y), C=1.0, gamma=1.0, n_iters=5000):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.n_iters = n_iters
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j], self.gamma)

        # Initialize alphas
        self.alpha = np.zeros(n_samples)
        # Gradient descent to optimize alphas
        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y[i] * (np.dot(self.alpha * y, K[i]) - self.b) < 1
                if condition:
                    self.alpha[i] += self.C * (1 - y[i] * (np.dot(self.alpha * y, K[i]) - self.b))
                    self.alpha[i] = max(0, min(self.alpha[i], self.C))
                # Update bias
                self.b = y[i] - np.dot(self.alpha * y, K[i])

        # Support vectors have non zero lagrange multipliers
        sv_mask = self.alpha > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.alpha = self.alpha[sv_mask]

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alpha, sv_y, sv in zip(self.alpha, self.support_vector_labels, self.support_vectors):
                s += alpha * sv_y * self.kernel(X[i], sv, self.gamma)
            y_pred[i] = s
        return np.sign(y_pred + self.b)

def gaussian_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

# Example usage
if __name__ == "__main__":
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.1, random_state=1)
    y[y == 0] = -1
    svm = KernelSVM(kernel=gaussian_kernel, C=1.0, gamma=2.0)
    svm.fit(X, y)

    # Create grid to evaluate model
    xx = np.linspace(-1.5, 1.5, 300)
    yy = np.linspace(-1.5, 1.5, 300)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm.predict(xy).reshape(XX.shape)

    # Plotting
    plt.contourf(XX, YY, Z, alpha=0.5, levels=np.linspace(Z.min(), Z.max(), 3), cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
    plt.title("Kernel SVM with Gaussian Kernel")
    # plt.show()
    plt.savefig("SVM.png")

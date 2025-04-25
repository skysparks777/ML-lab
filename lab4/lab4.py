#Lab 4: Logistic Regression using scikit-learn and Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#1. Create a binary classification dataset using scikit-learn’s make_classification() function.
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title('Binary Classification Dataset')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, )

#2. Train a Logistic Regression model using LogisticRegression from scikit-learn.
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_sklearn = clf.predict(X_test)

#3. Implement Logistic Regression from scratch using NumPy (Gradient Descent).
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

model_scratch = LogisticRegressionScratch()
model_scratch.fit(X_train, y_train)
y_pred_scratch = model_scratch.predict(X_test)

#/4. Evaluate both models using accuracy, confusion matrix, and classification report.
print("== Scikit-learn Logistic Regression ==")
print("Accuracy:", accuracy_score(y_test, y_pred_sklearn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_sklearn))
print("Classification Report:\n", classification_report(y_test, y_pred_sklearn))

print("\n== Logistic Regression from Scratch ==")
print("Accuracy:", accuracy_score(y_test, y_pred_scratch))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_scratch))
print("Classification Report:\n", classification_report(y_test, y_pred_scratch))

#5. Visualize the decision boundary for both models (if 2D
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)
    preds = np.array(preds).reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.5, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.title(title)
    plt.show()

plot_decision_boundary(X, y, clf, "Decision Boundary - scikit-learn")
plot_decision_boundary(X, y, model_scratch, "Decision Boundary - From Scratch")
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Plot the tree
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree on Iris Dataset")
plt.show()

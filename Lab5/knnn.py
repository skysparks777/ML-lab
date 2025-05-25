from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

X = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]  
y = [0, 0, 0, 1, 1, 1] 

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

new_point = [[5, 5]]

# Predict the class
prediction = knn.predict(new_point)
print(f"Predicted class for {new_point} is: {prediction[0]}")

# Plot training points (0 is red, 1 is blue)
for i, label in enumerate(y):
    color = 'red' if label == 0 else 'blue'
    plt.scatter(X[i][0], X[i][1], color=color)

# Plot the new point
new_color = 'red' if prediction[0] == 0 else 'blue'
plt.scatter(new_point[0][0], new_point[0][1], color=new_color, marker='4')

# ✨ Label the new point
plt.text(new_point[0][0] + 0.2, new_point[0][1] - 0.3, "Point to measure", fontsize=9)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN Example')
plt.grid(True)
plt.show()

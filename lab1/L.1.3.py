# 3. Matplotlib Scatter Plot
import matplotlib.pyplot as p
import numpy as np
# Generate two NumPy arrays of 50 random numbers each
x_scatter = np.random.rand(50) * 100
y_scatter = np.random.rand(50) * 100

# Create scatter plot as x and y axis
p.scatter(x_scatter, y_scatter, color='green', alpha=0.5)

# Add title and axis labels
p.title("Random Scatter Plot")
p.xlabel("X-axis")
p.ylabel("Y-axis")
p.show()

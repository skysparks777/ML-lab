#4. Matplotlib Histogram

import matplotlib.pyplot as p
import numpy as np

# Generate a NumPy array of 1000 random numbers from a normal distribution
data = np.random.randn(1000)

# Create histogram
p.hist(data, bins=30, color='purple', alpha=0.7)

# Add title and axis labels
p.title("Histogram of Normal Distribution")
p.xlabel("Value")
p.ylabel("Frequency")
p.show()

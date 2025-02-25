#2. Matplotlib Line Plots

import matplotlib.pyplot as p
import numpy as np
# Generate a NumPy array of 100 evenly spaced numbers between 0 and 2π
x = np.linspace(0, 2 * np.pi, 100)

# Calculate sine and cosine
sin_x = np.sin(x)
cos_x = np.cos(x)

# Plot the sine and cosine curves
p.plot(x, sin_x, label="Sine", color="blue")
p.plot(x, cos_x, label="Cosine", color="red")

# Add title, axis labels, and legend
p.title("Sine and Cosine Functions")
p.xlabel("X values")
p.ylabel("Function values")
p.legend()
p.show()

#1. NumPy Array Creation and Manipulation
import numpy as np

# Create a NumPy array of random integers between 1 and 100, with shape (10, 5)
arr = np.random.randint(1, 101, size=(10, 5))
print( arr)

# Calculate the mean, median, and standard deviation of each column
mean_col = np.mean(arr, axis=0)
print("Mean:", mean_col)
median_col = np.median(arr, axis=0)
print("Median:", median_col)
std_dev_col = np.std(arr, axis=0)
print("Standard Deviation:", std_dev_col)

# Find the maximum and minimum values in the entire array
max_val = np.max(arr)
min_val = np.min(arr)
print("Max:", max_val, "Min:", min_val)

# Slice the array to extract the first 3 rows and the last 2 columns
sliced_arr = arr[:3, -2:]
print("Sliced Array:\n", sliced_arr)

# Reshape the array into a 1D array
reshaped_arr = arr.flatten()
print("Reshaped Array:\n", reshaped_arr)

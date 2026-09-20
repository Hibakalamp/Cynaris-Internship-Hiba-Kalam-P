import numpy as np

# Arrays
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
c = np.array([[[1], [2]], [[3], [4]]])


print("Shape of 1D array:", a.shape)
print("Shape of 2D array:", b.shape)
print("Shape of 3D array:", c.shape)

# Operations
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print("Broadcast (x + 5):", x + 5)
print("Vectorised (x * y):", x * y)

# Matrix multiplication
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

print("Result Matrix Multiplication:\n", np.matmul(m1, m2))

# Load dataset
data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

# Columns
day = data[:, 0]
steps = data[:, 1]
calories = data[:, 2]

# Mean
print("Mean Steps:", np.mean(steps))
print("Mean Calories:", np.mean(calories))

# Standard Deviation
print("Std Steps:", np.std(steps))
print("Std Calories:", np.std(calories))

# Correlation
print("Correlation (Steps vs Calories):")
print(np.corrcoef(steps, calories))
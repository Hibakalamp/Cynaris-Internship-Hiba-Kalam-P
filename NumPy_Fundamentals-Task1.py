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

# Broadcasting
x = np.array([1, 2, 3])
y = x * 2

# Vectorised operations
print("Squared:", x ** 2)

# Matrix multiplication
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

print("Matrix product:", np.dot(m1, m2))

# Statistics
data = np.array([10, 20, 30, 40])

print("Mean:", np.mean(data))
print("Std:", np.std(data))
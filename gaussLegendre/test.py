import numpy as np

# Define the function to integrate
def f(x):
    return x**2  # Replace this with your function

# Number of Gauss points
n = 5

# Get Gauss-Legendre points (nodes) and weights
nodes, weights = np.polynomial.legendre.leggauss(n)

# Transform nodes to the interval [a, b]
a, b = 0, 1
transformed_nodes = 0.5 * (nodes + 1) * (b - a) + a
transformed_weights = 0.5 * (b - a) * weights

# Compute the integral
result = sum(f(x) * w for x, w in zip(transformed_nodes, transformed_weights))

print("Result of integration:", result)

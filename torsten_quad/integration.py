import pickle
from to_integrate import h
import numpy as np

'''
First integral 
'''
# open the points and weights
with open('radius.pkl', 'rb') as file:
    r = pickle.load(file)

with open('weight.pkl', 'rb') as file:
    w = pickle.load(file)

# perform the integration
first_integral = 0
counter = 0

for radius in r:
    first_integral = first_integral + w[counter]*h(radius) 
    counter = counter + 1

# check partial result
print("First integral: ", first_integral)

'''
Second integral
'''
# Define the function to integrate
def f(x):
    # Here we introduce the rational weight
    return h(x)*(1-x)/x # hopefully the integration weights will never include 0, otherwise this will be undefined

# Number of Gauss points
n = 5

# Get Gauss-Legendre points (nodes) and weights
nodes, weights = np.polynomial.legendre.leggauss(n)

# Transform nodes to the interval [a, b]
a, b = 0, 1
transformed_nodes = 0.5 * (nodes + 1) * (b - a) + a
transformed_weights = 0.5 * (b - a) * weights

# Compute the integral
second_integral = sum(f(x) * w for x, w in zip(transformed_nodes, transformed_weights))

# check partial result
print("Second integral:", second_integral)

'''
Final result
'''
result = first_integral + second_integral
print()
print("Final result: ", result)

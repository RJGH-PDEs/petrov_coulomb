# imports
import pickle
import numpy as np

# integration 
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev

'''
On this file we try to create a list that
contains all the points and weights to perform
the 6 dimensional integration for the Landau 
collision operator in the Coulumb interaction
'''

# open the special quadrature
with open('../special_quad/special_quad.pkl', 'rb') as file:
    special_quadrature = pickle.load(file)

# print it
print("Special quadrature: ")
print(special_quadrature)

'''
choose the integration order here
'''
n_laguerre = 3
n_lebedev = 5

# extract the coefficients
alpha = 1/2
x,w_r = roots_genlaguerre(n_laguerre, alpha, False)
lag = []
for point, weight in zip(x, w_r):
    '''
    we change variables 
    '''
    new_point  = np.sqrt(2 * point)
    new_weight = np.pi*4*np.sqrt(2)*weight
    # append
    lag.append([new_point, new_weight])

print("Radial integration: ")
print(lag)

# build library
leblib = PyLebedev()
s,w_spher = leblib.get_points_and_weights(n_lebedev)
leb = []
for p, w in zip(s,w_spher):
    leb.append([p, w])

print("Spherical integration:")
print(leb)


'''
Tensorize these
'''
# tensorized = [[(x, y) for y in lag] for x in leb]
tensorized = []
for radial in lag:
    for ang in leb:
        tensorized.append([radial, ang])
print()
print("tensorized: ")
print(tensorized)

# Can we test this now?
partial_sum = 0
for quad in tensorized:
    print()
    print(quad)
    print()
    # extract the points
    radial_point  = quad[0][0]
    radial_weight = quad[0][1]
    print(radial_point, radial_weight)
    angular_point   = quad[1][0]
    angular_weight  = quad[1][1]
    print(angular_point, angular_weight)

    # partial sum
    partial_sum = partial_sum + angular_weight*radial_weight


print(partial_sum)
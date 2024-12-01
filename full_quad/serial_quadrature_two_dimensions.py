# imports
import pickle
import numpy as np

# integration 
from scipy.special import roots_genlaguerre
from pylebedev import PyLebedev

'''
Spherical coordinates
'''
# radius    
def radius(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

# theta 
def theta(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)

    if r == 0:
        return 0
    else:
        return np.arccos(z/ r)
# phi    
def phi(x, y):
    r = np.sqrt(x**2 + y**2)

    if r == 0:
        return 0
    elif y == 0:
        return np.arccos(x/r) # new discovery, this might be wrong
    else:
        return np.sign(y)*np.arccos(x/r)


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
    new_point  = np.sqrt(point) # note that this is different from the previous case
    new_weight = weight/2
    # append
    lag.append([new_point, new_weight])

print("Radial integration: ")
print(lag)

# build library
leblib = PyLebedev()
s,w_spher = leblib.get_points_and_weights(n_lebedev)
leb = []
sum = 0
for p, w in zip(s,w_spher):
    leb.append([p, np.pi*4*w])
    x = p[0]
    y = p[1]
    z = p[2]
    sum = sum + w*(np.sin(theta(x, y, z)))


print("Spherical integration:")
print(leb)


'''
Tensorize these
'''
# tensorized = [[(x, y) for y in lag] for x in leb]
tensorized = []
for radial in special_quadrature:
    for ang in leb:
        tensorized.append([radial, ang])
print()
print("tensorized: ")
print(tensorized)

'''
Test the function
'''
# test function
def to_integrate_test(r, t, p):
    return r**3*(r*np.cos(t))**2

# numerical integration
partial_sum = 0
for quad in tensorized:
    # radial quadrature
    radial_point  = quad[0][0]
    radial_weight = quad[0][1]
    print("radial quadrature")
    print(radial_point, radial_weight)

    # angular quadrature
    angular_point   = quad[1][0]
    angular_weight  = quad[1][1]
    print("angular quadrature")
    print(angular_point, angular_weight)

    # cartesian quadrature point on the sphere
    r = radial_point
    x = angular_point[0]
    y = angular_point[1]
    z = angular_point[2]

    # perform the partial sum
    partial_sum = partial_sum + angular_weight*radial_weight*to_integrate_test(r, theta(x, y, z), phi(x, y))

print(partial_sum)
print(4*np.pi*sum*(np.pi/2)**(1/2))
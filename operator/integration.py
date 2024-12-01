import pickle
import numpy as np
import time 
# import quadrature unpacker
from unpack_quad import unpack_quadrature
# import integrand
from integrand import integrand
# import to produce the weight pieces
from derivatives import weight_new


'''
# open quadrature
with open('quadrature.pkl', 'rb') as file:
    quadrature = pickle.load(file)

# test functions
def f(r, t, p): # integrand in p
    return 1
def g(r, t, p): # integrand in u
    return r**3

# numerical integration
sum = 0
for q in quadrature:
    # unpack quadrature 
    weight, r_p, t_p, p_p, r_u, t_u, p_u = unpack_quadrature(q)
    # sample the function
    sample = f(r_p, t_p, p_p)*g(r_u, t_u, p_p)
    # update sum
    sum = sum + weight*sample

# print result
print(sum)
'''

# The Landau Operator
def operator(select, l, m, u, gradient, proj, hess):
    # open quadrature
    with open('quadrature.pkl', 'rb') as file:
        quadrature = pickle.load(file)

    print(len(quadrature))

    # numerical integration
    sum = 0
    for q in quadrature:
        # unpack quadrature 
        weight, r_p, t_p, p_p, r_u, t_u, p_u = unpack_quadrature(q)
        # sample the function
        sample = integrand(select, l, m, u, gradient, proj, hess, r_p, t_p, p_p, r_u, t_u, p_u)
        # update sum
        sum = sum + weight*sample

    return sum

# The main function
def main():
    # parameters for the weight
    k = 2
    l = 0
    m = 0

    # produce the weight pieces
    u, gradient, proj, hess = weight_new(k, l, m)

    # select the test functions
    kp = 0
    lp = 0
    mp = 0

    kq = 0
    lq = 0
    mq = 0

    # package on a vector
    select = [kp, lp, mp, kq, lq, mq]

    # Record the start time
    start_time = time.time()

    # compute the operator
    print(operator(select, l, m, u, gradient, proj, hess))

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")

   
# Call the main function
if __name__ == "__main__":
    main()


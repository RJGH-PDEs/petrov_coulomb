# imports
import pickle
import numpy as np
import time 
# import quadrature unpacker
from unpack_quad import unpack_quadrature
# import integrand
from integrand import integrand
# import to produce the weight pieces
from derivatives import weight_new

# loads the quadrature
def load_quad():
    with open('../full_quad/quadrature.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# The Landau Operator
def operator(select, l, m, u, gradient, proj, hess, quadrature):
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

# version of the operator that will be used for parallelization
def operator_parallel(select, shared):
    '''
    upack the shared data
    ''' 
    # quadrature 
    quadrature  = shared[0]
    # pieces from the weight
    u           = shared[1]
    gradient    = shared[2]
    proj        = shared[3]
    hess        = shared[4] 
    # test function (for the weight)
    k           = shared[5]
    l           = shared[6]
    m           = shared[7] 
    
    # compute the Landau operator
    result = operator(select, l, m, u, gradient, proj, hess, quadrature)
    # print the results
    print('Select: ', select, ', RESULT: ', result)
    # return results with all the information
    return [[k,l,m], select, result]

# operator
def operator_test(weight_param, quad):
    print("starting job for weight: ", weight_param)

    # unpack the choice of parameters
    k = weight_param[0]
    l = weight_param[1]
    m = weight_param[2]
    
    # produce the weight pieces
    u, gradient, proj, hess = weight_new(k, l, m)

    # select the test functions
    kp = 0
    lp = 0
    mp = 0

    kq = 1
    lq = 0
    mq = 0

    # package on a vector
    select = [kp, lp, mp, kq, lq, mq]

    # Record the start time
    start_time = time.time()

    # compute the operator
    result = operator(select, l, m, u, gradient, proj, hess, quad)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print('selection: ', k, m, l, ', RESULT: ', result, ', time ', elapsed_time)
    return result

# The main function
def main():
    # parameters for the weight
    k = 2
    l = 1
    m = 1
    param = [k, l, m]

    # load the quadrature
    quad = load_quad()

    # print the size of the quadrature
    print("quadrature length: ", len(quad))

    # test the operator
    operator_test(param, quad)

    
   
# Call the main function
if __name__ == "__main__":
    main()

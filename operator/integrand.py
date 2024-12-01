import numpy as np
# import weight elements
from weight_evaluator import weight_evaluator_u
from derivatives import weight_new
# import change of variables
from change_var import change_variables
# import test functions
from test_func import f_integrated

def test():
    # choice of test function
    k = 1
    l = 0
    m = 0

    # p point
    rp = 4
    tp = np.pi/6
    pp = np.pi/3

    # q point
    ru = 1
    tu = np.pi/6
    pu = np.pi/3

    # compute the pieces
    u, gradient, proj, hess = weight_new(k, l, m)

    # evaluate the weight
    w = weight_evaluator_u(l, m, u, gradient, proj, hess, rp, tp, pp, ru, tu, pu)
    print("weight result:", w)

# computes the inner product on spherical coordinates 
def inner_product(rp, tp, pp, ru, tu, pu):
    # convert to cartesian
    p = np.array([np.sin(tp)*np.cos(pp), np.sin(tp)*np.sin(pp), np.cos(tp)])
    u = np.array([np.sin(tu)*np.cos(pu), np.sin(tu)*np.sin(pu), np.cos(tu)])

    # contract
    result = p @ u
    result = result*rp*ru

    return result

# the integrand
def integrand(select, l, m, u, gradient, proj, hess, rp, tp, pp, ru, tu, pu):
    # evaluate the weight
    result = weight_evaluator_u(l, m, u, gradient, proj, hess, rp, tp, pp, ru, tu, pu)

    # mixed exponential
    result = result*np.exp(inner_product(rp, tp, pp, ru, tu, pu))

    '''
    Now we need to compute the basis functions,
    one at p, the other at p - u, i.e. at q, which
    means that i need to compute q first
    '''

    # compute the change of variables
    rq, tq, pq = change_variables(rp, tp, pp, ru, tu, pu)

    # sample the test functions
    result = result*f_integrated(select, rp, tp, pp, rq, tq, pq)

    return result

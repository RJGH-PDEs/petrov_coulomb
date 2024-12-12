from derivatives import weight_new, weight_evaluator
from to_numpy import to_numpy, weight_evaluator_numpy
from change_var import change_variables
import numpy as np

# new evaluation in terms of u
def weight_evaluator_u(l, m, u, grad, projection, hessian, rp, tp, pp, ru, tu, pu):
    # first we change coordinates to q
    rq, tq, pq = change_variables(rp, tp, pp, ru, tu, pu)

    '''
    print("q coordinates:")
    print(rq, tq, pq)
    '''

    # then we call the previous weight in terms of q
    return (-1)*weight_evaluator(l, m, u, grad, projection, hessian, rp, tp, pp, rq, tq, pq) # <-- we multiply by -1, taking into account the change of vars

def np_weight_evaluator_u(l, m, u, grad, projection, hessian, rp, tp, pp, ru, tu, pu):
    # first we change coordinates to q
    rq, tq, pq = change_variables(rp, tp, pp, ru, tu, pu)
    # return the evaluation of the weight
    return (-1)*weight_evaluator_numpy(l, m, u, grad, projection, hessian, rp, tp, pp, rq, tq, pq)


# main function
def main():
    # choice of test function
    k = 0
    l = 2
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

    # compute the pieces again
    u, gradient, proj, hess = to_numpy(k, l, m)
    w = np_weight_evaluator_u(l, m, u, gradient, proj, hess, rp, tp, pp, ru, tu, pu)
    print("weight result:", w)

if __name__ == '__main__':
    main()
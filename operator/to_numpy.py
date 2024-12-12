# sympy and numpy
import sympy as sp
import numpy as np
import time

# compute the derivatives
from derivatives import weight_new
from derivatives import spher_const

def to_numpy(weight_param):
    # symbols
    r = sp.symbols('r')
    t = sp.symbols('t')
    p = sp.symbols('p')
    # more symbols
    rp = sp.symbols('rp')
    tp = sp.symbols('tp')
    pp = sp.symbols('pp')

    rq = sp.symbols('rq')
    tq = sp.symbols('tq')
    pq = sp.symbols('pq')
 
    # unpack the choice of parameters
    k = weight_param[0]
    l = weight_param[1]
    m = weight_param[2]
    
    # produce the weight pieces
    u, gradient, proj, hess = weight_new(k, l, m)

    # gradient and hessian
    grad_np = sp.lambdify([r, t, p], gradient, "numpy")
    hess_np = sp.lambdify([r, t, p], hess, "numpy")

    # relative position and projection
    u_np = sp.lambdify([rp, tp, pp, rq, tq, pq], u, "numpy")
    p_np = sp.lambdify([rp, tp, pp, rq, tq, pq], proj, "numpy")
    # test it
    '''
    Comparison
    a = 3
    b = np.pi/2
    c = 0

    values={"r":a,"t":b,"p":c}
    tic = time.time()
    result = gradient.subs(values)
    toc = time.time()
    print("time: ", tic - toc, ", result: ", result)
    tic = time.time()
    result = grad_np(a, b, c)
    toc = time.time()
    print("time: ", tic - toc, ", result: ", result)

    '''
    return u_np, grad_np, p_np, hess_np

# weight evaluator
def weight_evaluator_numpy(weight_param, u, grad, projection, hessian, pt):
    
    # unpack weight parameters
    l = weight_param[1]
    m = weight_param[2]
    # evaluation point 
    rp = pt[0]
    tp = pt[1]
    pp = pt[2]
    rq = pt[3]
    tq = pt[4]
    pq = pt[5]
    
    # grad difference 
    gradDiff = np.ravel(grad(rp, tp, pp) - grad(rq, tq, pq))
    # relative position
    rel_pos  = np.ravel(u(rp, tp, pp, rq, tq, pq))

    inner = -2*(gradDiff@rel_pos)
    
    # sum of hessians
    sumhess = hessian(rp, tp, pp) + hessian(rq, tq, pq)
    # projection
    proj    = projection(rp, tp, pp, rq, tq, pq)

    # compute the contraction
    # contraction = (np.trace(np.matmul(proj,sumhess)))/2
    contraction = (np.sum(proj*sumhess))/2
    
    return (inner + contraction)*spher_const(l, m)

# The main function
def main():
    # parameters for the weight
    k = 2
    l = 2
    m = -2
    weight = [k, l, m]

    u_np, grad_np, p_np, hess_np = to_numpy(weight)
    
    a = 4
    b = np.pi/6
    c = np.pi/3

    d = 1
    e = np.pi/3
    f = np.pi/6

    point = [a, b, c, d, e, f]


    tic = time.time()
    s = weight_evaluator_numpy(weight, u_np, grad_np, p_np, hess_np, point)
    toc = time.time()
    print("time ", toc - tic)
    print("result: ", s)
# Call the main function
if __name__ == "__main__":
    main()
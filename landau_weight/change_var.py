import numpy as np
from derivatives import rel_pos 

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

# change of variables
def change_variables(rp, tp, pp, ru, tu, pu):
    '''
    Given p and u points, we compute p and q points, 
    so we can reuse the old code... this should work
    '''
    q = rel_pos(rp, tp, pp, ru, tu, pu)

    # now we have q in cartesian, but we need it on spherical coordinates
    rq = radius(q[0], q[1], q[2]) 
    tq = theta(q[0], q[1], q[2]) 
    pq = phi(q[0], q[1])

    return [rq, tq, pq]

# main function
def main():
    k = 2
    l = 2
    m = -2

    rp = 4
    tp = np.pi/6
    pp = np.pi/3

    ru = 1
    tu = np.pi/3
    pu = np.pi/6

    print("q coordinates:")
    print(change_variables(rp, tp, pp, ru, tu, pu))

if __name__ == '__main__':
    main()
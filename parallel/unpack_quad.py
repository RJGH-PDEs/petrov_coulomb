import numpy as np    
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


def unpack_quadrature(quad):
    # radial quadrature
    r_p = quad[0][0]
    w_p = quad[0][1]

    # angular quadrature
    ang_p   =  quad[1][0]
    ang_w_p = quad[1][1]

    # special radial quadrature
    r_u = quad[2][0]
    w_u = quad[2][1]
    
    # angular quadrature for u
    ang_u   = quad[3][0]
    ang_w_u = quad[3][1] 

    # cartesian quadrature point on the sphere
    x_p = ang_p[0]
    y_p = ang_p[1]
    z_p = ang_p[2]

    x_u = ang_u[0]
    y_u = ang_u[1]
    z_u = ang_u[2]

    # exctract angular variables
    t_p = theta(x_p, y_p, z_p)
    p_p = phi(x_p, y_p)

    t_u = theta(x_u, y_u, z_u)
    p_u = phi(x_u, y_u)

    # full weight 
    weight = w_p*ang_w_p*w_u*ang_w_u 
    
    '''
    Now we need to deal how to return this 
    '''
    return [weight, r_p, t_p, p_p, r_u, t_u, p_u]

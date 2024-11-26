import pickle
from unpack_quad import unpack_quadrature

# open quadrature
with open('quadrature.pkl', 'rb') as file:
    quadrature = pickle.load(file)

# test functions
def f(r, t, p): # integrand in p
    return 1
def g(r, t, p): # integrand in u
    return r

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
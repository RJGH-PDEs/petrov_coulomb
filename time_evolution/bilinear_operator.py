import numpy as np
import pickle

# finally, the biliear Landau operator
def landau(sparse_op, f, result):
    '''
    computes Q(f, f) where f is given as a vector.
    the sparse operator is applied like: (f^T)sparse_op(f),
    producing a vector, whose entries are put into result vector
    '''
    i = 0
    # compute 
    for sm in sparse_op:
        # compute the rayleigh form
        r = f @ (sm.dot(f))
        # print("matrix ", sm.nnz)
        # print("result: ", r)
        
        # store it 
        result[i] = r
        i = i + 1

# update: copies b into a
def update(a, b):
    i = 0

    for val in b:
        a[i] = val
        i = i + 1

if __name__ == "__main__":
    # open the sparse operator
    with open('../sparse_operator/sparse_operator.pkl', 'rb') as file:
        so = pickle.load(file)

    # check individual entries
    # p = 18
    # q = 15
    # w = 24
    # sparse_matrix = so[w]
    # print(sparse_matrix[p,q])
    # print(so)

    result = np.zeros(27)    

        # iterate over all basis vectors
    for i in range(27):
        print('iteration: ', i)

        # we now test it 
        f = np.zeros(27)
        f[i] = 1

        # apply the operator
        landau(so, f, result)

        print('norm of result: ', np.linalg.norm(result))
        print()
        '''
        for number in result:
            print(number)

        print()
        '''


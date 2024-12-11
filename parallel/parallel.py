import multiprocessing
import os
import time
import pickle

# import pieces from operator
from landau import load_quad, operator_parallel
# import to produce the weight pieces
from derivatives import weight_new 


# create iterable
def create_param_iterable(n):
    param = []
    
    # iterate over all p
    for kp in range(0, n):
        for lp in range(0, n):
            for mp in range(-lp, lp+1):
                # iterate over all q    
                for kq in range(0, n):
                    for lq in range(0, n):
                        for mq in range(-lq, lq+1): 
                            # create the select
                            select = [kp, lp, mp, kq, lq, mq]
                            param.append(select)

    return param

# given a weight iterates over different trial functions
def trial_iterator(data, r):
    # Create a manager for shared data
    manager = multiprocessing.Manager()
    shared_data = manager.list(data)

    # Define the list of parameters
    params = create_param_iterable(n)

    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use starmap to pass the shared data to each worker
        results = pool.starmap(operator_parallel, [(select, shared_data) for select in params])

    r.append(results)

# produce collision matrix
def weight_iteration(n, r):
    # Load data once 
    quad = load_quad()
    print("quadrature size: ", len(quad))

    # iterate over all possible weights
    for k in range(0, n):
        for l in range(0, n):
            for m in range(-l, l+1):

                # print the weight
                print("weight on: ", k, l, m)

                # produce required pieces for the weight
                u, g, p, h = weight_new(k, l, m)

                # package shared data (sd)
                sd = [quad, u, g, p, h, k, l, m]
                
                # compute it and time it
                start = time.time()
                trial_iterator(sd, r)
                end = time.time()

                # save the partial result
                with open('../results/operator.pkl', 'wb') as file:
                    pickle.dump(r, file)

                # Calculate elapsed time
                elapsed_time = end - start
                print(f"Elapsed time: {elapsed_time:.6f} seconds") 
                print()

# main function
if __name__ == "__main__":
    # select the degrees of freedom
    n = 3

    # results to be stored here
    r = []

    # compute the tensor
    weight_iteration(n, r)

    # print the result
    print(r)

    # save the result
    with open('../results/operator.pkl', 'wb') as file:
        pickle.dump(r, file)
import multiprocessing
import os
import time
import pickle

# import pieces from operator
from landau import load_quad, operator_recompute
# import to produce the weight pieces
from derivatives import weight_new 

# splits the list
def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

# create iterable
def load_non_zeros():
    # Loading the data
    with open('../results/non_zeros.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    '''
    for element in loaded_data:
        print(element[-1])
    '''
    return loaded_data

# given a weight iterates over different trial functions
def parallel_iterator(params, data):
    # Create a manager for shared data
    manager = multiprocessing.Manager()
    shared_data = manager.list(data)

    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use starmap to pass the shared data to each worker
        results = pool.starmap(operator_recompute, [(p, shared_data) for p in params])

    return results

# recompute the collision matrix
def recompute(r):
    # batch size
    n = 600

    # Load data once 
    quad = load_quad()
    print("quadrature size: ", len(quad))

    # Define the list of parameters
    non_zeros = load_non_zeros()

    # split it 
    split_non_zeros = split_list(non_zeros, n)
    
    # call the iterator
    for sublist in split_non_zeros:
        results = parallel_iterator(sublist, quad)
        
        # save the partial result
        r.append(results)
        with open('../results/recomputed.pkl', 'wb') as file:
            pickle.dump(r, file)

    '''
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
                end = time.time()

                # save the partial result
                with open('../results/operator.pkl', 'wb') as file:
                    pickle.dump(r, file)

                # Calculate elapsed time
                elapsed_time = end - start
                print(f"Elapsed time: {elapsed_time:.6f} seconds") 
                print()
    '''

# main function
if __name__ == "__main__":
    # results to be stored here
    r = []
    
    # recompute
    recompute(r)

    # print the result
    print(r)

    # save the result
    with open('../results/recomputed.pkl', 'wb') as file:
        pickle.dump(r, file)
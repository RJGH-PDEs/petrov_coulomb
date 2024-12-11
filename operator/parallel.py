import multiprocessing

from integration import load_quad, operator_test

# create iterable
def create_param_iterable():
    n = 3
    param = []
    # iterate over all possible weights
    for k in range(0, n):
        for l in range(0, n):
            for m in range(-l, l+1):
                p = [k, l, m]
                param.append(p)

    return param

if __name__ == "__main__":
    # Load data once in the main process
    data = load_quad()
    print("quadrature size: ", len(data))

    # Create a manager for shared data
    manager = multiprocessing.Manager()
    shared_data = manager.list(data)

    # Define the list of parameters
    params = create_param_iterable()

    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use starmap to pass the shared data to each worker
        results = pool.starmap(operator_test, [(param, shared_data) for param in params])

    print("Results:", results)

import pickle
import numpy as np

# Loading the data
with open('operator.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Find the location of the non-zeros
non_zeros = []
counter = 0
for slice in loaded_data:
    for d in slice:
        # print(d[2])
        if np.abs(d[2]) > 10e-10:
            non_zeros.append(d)
            counter = counter + 1

# print(non_zeros)
print(counter)

# Save 
with open('non_zeros.pkl', 'wb') as file:
    pickle.dump(non_zeros, file)
    print("data has been saved")

'''
counter = 0
slice = loaded_data[26]
for d in slice:
    if np.abs(d[2]) > 10:
        print(d)
        non_zeros.append(d)
        counter = counter + 1

print(counter)
'''
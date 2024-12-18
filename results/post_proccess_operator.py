import pickle
import numpy as np

# Loading the data
with open('recomputed.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

difference = []
for batch in loaded_data:
    for element in batch:
        difference.append(element[2] - element[3])

print("max difference: ", np.max(np.abs(difference)))
'''
# Loading the data
with open('operator_old.pkl', 'rb') as file:
    old_op = pickle.load(file)

difference = []
i = 0
for slice in loaded_data:
    j = 0
    for d in slice:
        difference.append(d[2] - old_op[i][j][2])
        # print(d[2] - loaded_data[i][j][2])
        j = j + 1

    i = i + 1

print("max difference: ", np.max(np.abs(difference)))
print(len(loaded_data[0]))

for slice in loaded_data:
    for element in slice:
        print(element)

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

counter = 0
slice = loaded_data[26]
for d in slice:
    if np.abs(d[2]) > 10:
        print(d)
        non_zeros.append(d)
        counter = counter + 1

print(counter)
'''
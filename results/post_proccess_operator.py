import pickle
import numpy as np

# Loading the older data
with open('operator_957.pkl', 'rb') as file:
    old_computation= pickle.load(file)

# Loading the newer data
with open('operator_1069.pkl', 'rb') as file:
    new_computation = pickle.load(file)

# tolerance for the discrepancie
tol = 1
# count the number where discrepancie bigger than tol
count = 0
# will store all the differences
difference = []
i = 0
for slice in old_computation:
    j = 0
    for d in slice:
        difference.append(d[2] - new_computation[i][j][2])
        discrepancie = d[2] - new_computation[i][j][2]
        if np.abs(discrepancie) > tol:
            # print(discrepancie)
            count = count + 1
        j = j + 1

    i = i + 1

print("max difference: ", np.max(np.abs(difference)))
print("number where discrepancie is big: ", count)

'''
Find the location of the non-zeros
''' 
non_zeros = []
counter = 0
for slice in new_computation:
    for d in slice:
        # print(d[2])
        if np.abs(d[2]) > 10e-10:
            non_zeros.append(d)
            counter = counter + 1

# print(non_zeros)
print('proportion of non-zeros in new computation: ', 100*counter/(27**3), '%')

'''
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

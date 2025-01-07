import pickle
import numpy as np

# Loading the older data
with open('operator_13913.pkl', 'rb') as file:
    old_computation= pickle.load(file)

# Loading the newer data
with open('operator_13913_nonegative.pkl', 'rb') as file:
    new_computation = pickle.load(file)

# tolerance for the discrepancie
tol = 1
# count the number where discrepancie bigger than tol
count = 0
# will store all the differences
difference = []

# iterate over all the entries 
i = 0
for slice in old_computation:
    j = 0
    for d in slice:
        # compute the difference
        old = d[2]
        new = (-1)*new_computation[i][j][2]
        diff = old - new
        print('old: ', old , ' new: ', new, ' difference: ', diff)
        difference.append(diff)
        '''        
        if np.abs(diff) > tol and np.abs(old) > 0.1:
            # compute the percentage difference
            percent = 100*diff/d[2]

            if np.abs(percent) > 10:
                # print('old: ', d[2], ' new: ', new_computation[i][j][2], ' percent: ', percent)
                print(d, new_computation[i][j])
                # store it
                difference.append(percent)
                count = count + 1
        '''
        j = j + 1

    i = i + 1

print("max difference: ", np.max(np.abs(difference)))
# print("number where discrepancie is big: ", count)

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

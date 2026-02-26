import numpy as np

# slicing the array
data = np.array([1,2,3,4,5])

print("print the whole matrix:", data[:])
print("print the last two digit :", data[-2:])

e = np.array([1,2,3])
f = np.array([4,5,6])

g = np.concatenate((e,f))

print("g =", g)

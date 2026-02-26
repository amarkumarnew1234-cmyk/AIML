import pandas as pd      # used for tabular data
import numpy as np       # used to handle array related data
import matplotlib.pyplot as plt   # data plotting
import seaborn as sns    # data visualization


# -------------------------------
# 1️⃣ Slicing the array
# -------------------------------

data = np.array([1,2,3,4,5])

print("print the whole matrix:", data[:])
print("print the last two digit :", data[-2:])


e = np.array([1,2,3])
f = np.array([4,5,6])

g = np.concatenate((e,f))

print("g =", g)


# -------------------------------
# 2️⃣ Arithmetic Operations
# -------------------------------

a = np.array([7,5,9])
print("a=", a)

b = np.array([1,2,3])
print("b=", b)

c = np.add(a,b)
print("c=", c)

d = np.subtract(a,b)
print("d=", d)

e = np.multiply(a,b)
print("e=", e)

f = np.divide(a,b)
print("f=", f)


# -------------------------------
# 3️⃣ arange + subtraction
# -------------------------------

a = np.array([20,30,40,50])

b = np.arange(4)
print("b=", b)

c = a - b
print("c=", c)


# -------------------------------
# 4️⃣ Power Operation
# -------------------------------

print("b**2 =", b**2)

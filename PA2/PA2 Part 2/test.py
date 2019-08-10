import bm_classify as bm
import numpy as np


x = [1, 2, 3]
x = 2
print(x)
x = np.array(x)
print(x)

print(np.add(x, 1))
x = x + 1
print(x)
print(np.divide(1, x))
x = 1 / x
print(x)

print(bm.sigmoid(1))

y = np.array([1, 2, 3])
z = 4

print(y)
print(z)
y = np.append(y, z)
print(y)
print()

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
ones = np.ones(len(a))
print(ones)

ap = np.column_stack((a, ones))
print(ap)

print(2 * ones)
print()

w = np.array([1, 1, 1])
D = len(w)
print(w)
wp = np.append(w, 0)
print(w)

w = wp[:D]
b = wp[D]

print(w)
print(b)


z = 4
a = np.array([1, 2])

print(np.append(4, a))
print()

t = np.array([[2, 2], [3, 4]])
print(t)
print(np.trace(t))
print()


a = np.array([1, 2, 3, 4])
print(a)
print(a.shape)
a = a.reshape(len(a), 1)
print(a)
print(a.shape)
print(a.T)
print((np.transpose(a)).shape)

x = np.atleast_2d(a)
print(x)
print(x.shape)

print()

D = 2
w = np.zeros((D,1))
print(w.shape)

print()

a = np.array([[2, 2, 3], [1, 1, 2]])
print(a)
y = np.array([2, 3])
print(y)

c = (a.T * y).T
print(c)

print(c.T)
print(np.sum(c, axis = 0))
print(np.sum(c.T))
print()


a = np.array([-2,4,3,-1,2])
print(a)
positive_indicator = np.int64(a <= 0)
print(positive_indicator)
print()
print()

a = np.array([[2, 1], [1, 1], [2, 3]])
print(a)
y = np.array([2, 3, 4])
print(y)

print(a.shape)
print(y.shape)
print((a.T + y).T)

print(a + y[:, np.newaxis])

print()
print()

w = np.array([[0, 0], [0, 0], [0, 0]])
C, D = w.shape

print(w.shape)
print(w)

b = np.ones(C)

c = np.column_stack((w,b))

print(c)
print()

print(c[:,:D])

print(c[:,D])
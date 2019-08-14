import numpy as np

a = [0, 1]
b = [1, 2]
c = [3, 4]

A = [a, b, c]
#print(A)

#print(np.random.choice(3, 2))

X = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
Y = np.array([[1,1,1], [2,2,2]])

#print(X)
#print(X.shape)

y = np.array([3, 3, 3])
#print(y)
#print(y.shape)

#print(X - y)

a = np.array(range(5, 10))
#print(a)
b = np.array(range(2, 6))
#print(b)

#res = a[np.newaxis, :] - b[:, np.newaxis]
#print(res)

print(X)
print(Y)

#res = np.subtract.inner(X, Y)
#print(res)

#print(res.shape)

#res = np.einsum('ijk,ijk->ij', X[:,None,:] - Y[None,:,:], X[:,None,:] - Y[None,:,:])
#for features in data:
    #distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]

m = X[np.random.randint(X.shape[0], size=2)]
print("m:")
print(m)

#distances = X - m[:, np.newaxis]
distances = np.sqrt(np.sum((X - m[:, np.newaxis]) ** 2, axis=2))
print(distances)
Z = np.argmin(distances, axis=0)

print(Z)
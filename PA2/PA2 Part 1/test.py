from linear_regression import mapping_data
import numpy as np

X = [1, 2, 3], [4, 5, 6]
X = np.array(X)
print(X)

mapped_x = mapping_data(X, 2)

print(mapped_x)
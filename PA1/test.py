# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

import numpy as np
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance
from utils import f1_score
from utils import NormalizationScaler, MinMaxScaler
from utils import Information_Gain
from utils import model_selection_with_transformation, model_selection_without_normalization
from hw1_knn import KNN

distance_funcs = {
    'euclidean': euclidean_distance,
    'gaussian': gaussian_kernel_distance,
    'inner_prod': inner_product_distance,
    'cosine_dist': cosine_sim_distance,
}

from data import data_processing
Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()

print(Xtrain[0])
print(ytrain[0])

p1 = [0, 0, 0]
p2 = [3, 3, 3]

print(euclidean_distance(p1, p2))


data =  [ [25.0, 3.8], [22.0,3.0] ]
l = [0,1]
mapped  = zip(data, l)
mapped = list(mapped)
for i in mapped:
    print(i)

m = {}
m[0] = 1
m[1] = 0
m[2] = 4
m[3] = 2

print(m)
m = sorted(m, key = m.get)
print(m)
a = m[0:2]
print(a)

print()
print("--------------------")
print("Normalization tests:")
print()

normalizer = NormalizationScaler()

normalize_test = [[3, 4], [1, -1], [0, 0]]
minmax_test = [[2, -1], [-1, 5], [0, 0]]
minmax_test2 = [[2, 2], [2, 3], [2, 2]]
minmax_test3 = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
minmax_test4 = [[2, 2], [2, 2], [2, 2]]
minmax_test5 = [[1, 100]]
labels = [0, 1, 1]

print()

print(normalizer(normalize_test))
print()

minmax = MinMaxScaler()
print("test 1")
print(minmax(minmax_test))
print()
print("test 2")
print(minmax(minmax_test2))
print()
print("test 3")
print(minmax(minmax_test3))
print()
print("test 4")
print(minmax(minmax_test4))
print()
print("test 5")
print(minmax(minmax_test5))
print() 

"""
scaling_classes = {
    'min_max_scale': MinMaxScaler,
    'normalize': NormalizationScaler,
}

for sc in scaling_classes:
    print(sc)
    print(scaling_classes[sc])
    print()

ig_test =  [[2, 5], [10, 3]]

Information_Gain(0.97, ig_test)

point1 = [1, 2, 3]
point2 = [3, 5, 7]

print(cosine_sim_distance(point1, point2))

"""

model_selection_without_normalization(distance_funcs, minmax_test, labels, minmax_test2, labels)

knn_dataset = [[0, 0], [4, 4], [2, 2], [3, 3], [1, 1], [5, 5]]
knn_labels = [1, 0, 1, 1, 0, 0]

knn_test = KNN(2, distance_funcs['euclidean'])
knn_test.train(knn_dataset, knn_labels)

print("KNN mapping:")
print(knn_test.mapping)

pred_test = [[3, 3]]
print(knn_test.predict(pred_test))
print()

print("k nearest neighbors:")
print(knn_test.get_k_neighbors(pred_test[0]))
print()

for d in knn_dataset:
    print(euclidean_distance(pred_test[0], d))

print()
print("---------------")
from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        #raise NotImplementedError
        self.mapping = list(zip(labels, features))
        return

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        #raise NotImplementedError

        ret = []
        for f in features:
            
            neighbors = self.get_k_neighbors(f)
            n = max(set(neighbors), key = neighbors.count)
            ret.append(n)

        return ret

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        #raise NotImplementedError
        """
        dist_map = {}
        for p in self.mapping:
            print(p[1])
            dist = self.distance_function(point, p[1])
            dist_map[p[0]] = dist
        
        points = sorted(dist_map, key = dist_map.get)
        print("points")
        print(points)
        print()
        ret = points[0:self.k]
        """
        """
        dist_mapping = {}
        for p in self.mapping:
            dist = self.distance_function(point, p[1])
            if len(dist_mapping) < self.k:
                dist_mapping[dist] = p[0]
            else:
                if dist < max(dist_mapping, key = int):
                    del dist_mapping[max(dist_mapping, key = dist_mapping.get)]
                    dist_mapping[dist] = p[0]

        ret = list(dist_mapping.values())
        """
        dist_mapping = []

        for p in self.mapping:
            dist = self.distance_function(point, p[1])
            if len(dist_mapping) < self.k:
                dist_mapping.append([dist, p[0]])
            else:
                dist_mapping.sort(key = lambda x: x[0])
                if dist < dist_mapping[self.k - 1][0]:
                    dist_mapping.pop(self.k - 1)
                    dist_mapping.append([dist, p[0]])
                    
        ret = []
        for m in dist_mapping:
            ret.append(m[1])

        return ret

if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)

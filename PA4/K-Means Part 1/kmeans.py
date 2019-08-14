import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    # raise Exception('Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    
    centers = []
    centers.append(generator.randint(n))
    
    for i in range(n_cluster - 1):
        distances = []
        for j, point in enumerate(x):
            nearestCenter = get_nearest_cluster_center(point, centers)
            #distances[j] = dist_squared(point, nearestCenter)
            distances.append(dist_squared(point, nearestCenter))
        #distSum = np.sum(distances)
        distSum = sum(distances)
        maxDist = 0
        mindex = 0
        for j, d in enumerate(distances):
            dist = d / distSum
            if dist > maxDist:
                maxDist = dist
                mindex = j

        centers.append(mindex)
    
    """
    centers = np.zeros(n_cluster)
    #distances = np.zeros(x.len)

    centers[0] = generator.randint(n)
    
    for i in range(n_cluster - 1):
        distances = []
        for j, point in enumerate(x):
            nearestCenter = get_nearest_cluster_center(point, centers)
            #distances[j] = dist_squared(point, nearestCenter)
            distances.append(dist_squared(point, nearestCenter))
        #distSum = np.sum(distances)
        distSum = sum(distances)
        maxDist = 0
        mindex = 0
        for j, d in enumerate(distances):
            dist = d / distSum
            if dist > maxDist:
                maxDist = dist
                mindex = j

        centers[i + 1] = mindex
    """

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers

def get_nearest_cluster_center(pt, centers):
    mindex = 0
    mdist = float("inf")
    for i, center in enumerate(centers):
        dist = dist_squared(pt, center)
        if dist < mdist:
            mindex = i
            dist = mdist
    #return mindex
    return centers[mindex]


def dist_squared(x, y):
    dist = np.square((x - y))
    return sum(dist)



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        print("N: " + str(N) + ", D: " + str(D))
        print("n_cluster: " + str(self.n_cluster))

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception('Implement fit function in KMeans class')
        
        mu_indices = self.generator.choice(N, self.n_cluster, replace = True)
        centroids = np.array([x[i] for i in mu_indices])

        y = 0

        for i in range(self.max_iter):
            distances = np.sqrt(np.sum((x - centroids[:, np.newaxis]) ** 2, axis=2))
            classifications = np.argmin(distances, axis=0)

            #R = np.array([classifications == c for c in range(self.n_cluster)])
            #R = np.array([range(self.n_cluster) == c for c in classifications])
    
            newCenters = np.array([np.mean(x[classifications == c], axis=0) for c in range(self.n_cluster)])
            if np.all(centroids - newCenters < self.e):
                self.max_iter = i + 1
                break
            centroids = newCenters
            y = classifications
            

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape

        print("N: " + str(N) + ", D: " + str(D))
        print("n_cluster: " + str(self.n_cluster))


        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception('Implement fit function in KMeansClassifier class')

        
        mu_indices = self.generator.choice(N, self.n_cluster, replace = True)
        centroids = np.array([x[i] for i in mu_indices])
        print("Means shape: " + str(centroids.shape))

        centroid_labels = np.zeros(self.n_cluster)

        for i in range(self.max_iter):
            distances = np.sqrt(np.sum((x - centroids[:, np.newaxis]) ** 2, axis=2))
            classifications = np.argmin(distances, axis=0)
    
            newCenters = np.array([np.mean(x[classifications == c], axis=0) for c in range(self.n_cluster)])
            if np.all(centroids - newCenters < self.e):
                self.max_iter = i + 1
                break
            centroids = newCenters
            #y = classifications
        
        byCluster = np.zeros((self.n_cluster, N))

        for i, c in enumerate(classifications):
            byCluster[c][y[i]] += 1

        centroid_labels = np.argmax(byCluster, axis=1)

        """

        R = np.array([(range(self.n_cluster) == c).astype(int) for c in y])

        voting = np.zeros((self.n_cluster, N))

        for centroid in voting:
            for p in y:
                centroid[p] += 1

        """




        # DONOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)
        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception('Implement predict function in KMeansClassifier class')

        distances = np.sqrt(np.sum((x - self.centroids[:, np.newaxis]) ** 2, axis=2))
        classifications = np.argmin(distances, axis=0)

        labels = [self.centroid_labels[i] for i in classifications]

 
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    #raise Exception('Implement transform_image function')

    """
    for dim in image:
        for rgb in dim:
            rgb = get_nearest_cluster_center(rgb, code_vectors)

    new_im = image
    """

    new_im = 0
    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im


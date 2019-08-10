import numpy as np

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features (D, N)
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (N, )
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression (D, )
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################

        """
        y[y == 0] = -1
        dw = np.zeros(D)
        db = 0
        for _ in range(max_iterations):
            for idx, x in enumerate(X):
                if (np.dot(w.T, x) + b) * y[idx] <= 0:
                    dw += step_size * y[idx] * x
                    db += step_size * y[idx]
            w = dw / N
            b = db / N
            
        

        y[y == -1] = 0
        """
    
        
        y[y == 0] = -1

        ones = np.ones(N)

        Xp = np.column_stack((X, ones))
        #Xp = np.column_stack((ones, X))
        wp = np.append(w, b)
        #wp = np.append(b, w)

        for _ in range(max_iterations):
            A = y * np.matmul(Xp, wp)
            ind = np.int64(A <= 0)
            wp += (step_size / N) * np.matmul(Xp.T, ind * y)
            
        w = wp[:D]
        b = wp[D]
        #w = wp[1:]
        #b = wp[0]

        y[y == -1] = 0
        


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################

        y[y == 0] = -1
        
        ones = np.ones(N)

        Xp = np.column_stack((X, ones))
        #Xp = np.column_stack((ones, X))
        wp = np.append(w, b)
        #wp = np.append(b, w)

        for _ in range(max_iterations):
            A = -1 * y * np.matmul(Xp, wp)
            ind = sigmoid(A)
            wp += (step_size / N) * np.matmul(Xp.T, ind * y)
            
        w = wp[:D]
        b = wp[D]
        #w = wp[1:]
        #b = wp[0]

        y[y == -1] = 0

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = z
    ############################################

    #value = np.divide(1, np.add(np.exp(-z), 1))
    value = 1 / (1 + np.exp(-z))
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################

        p = np.matmul(X, w) + b
        preds = np.int64(p > 0)


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################

        z = np.matmul(X, w) + b
        f = sigmoid(z)
        preds = np.int64(f >= 0.5)


    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds


#axis is 1 for GD and 0 for SGD
#works for both vectors and matrices
def softmax(z, axis):
    zp = np.atleast_2d(z)
    zp = zp - np.expand_dims(np.max(zp, axis = axis), axis)

    zp = np.exp(zp)
    denom = np.expand_dims(np.sum(zp, axis = axis), axis)

    return zp / denom


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################

        ones = np.ones(N)
        Xp = np.column_stack((X, ones))
        wp = np.column_stack((w, b))

        yp = np.zeros((N, C))
        yp[np.arange(N), y] = 1

        for _ in range(max_iterations):
            n = np.random.choice(N)
            x_n = Xp[n]
            y_n = yp[n]
            z = np.matmul(wp, x_n)

            sm = softmax(z, 1) - y_n

            sm = np.reshape(sm, (C, 1))
            x_n = np.reshape(x_n, (D+1, 1))
            grad = np.matmul(sm, x_n.T)

            wp -= step_size * grad


        w = wp[:,:D]
        b = wp[:,D]


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################

        ones = np.ones(N)
        Xp = np.column_stack((X, ones))
        wp = np.column_stack((w, b))

        yp = np.zeros((N, C))
        yp[np.arange(N), y] = 1

        for _ in range(max_iterations):
            z = np.matmul(wp, Xp.T)
            sm = softmax(z, 0)
            sm -= yp.T
            grad = np.matmul(sm, Xp)

            wp -= (step_size / N) * grad


        w = wp[:,:D]
        b = wp[:,D]
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################

    """
    for i in range(N):
        m = np.matmul(w, X[i]) + b
        preds[i] = np.argmax(m)
    """

    m = np.matmul(X, w.T)
    mp = m + b
    #n = np.add(m, b)
    preds = np.argmax(mp, axis = 1)


    assert preds.shape == (N,)
    return preds




        
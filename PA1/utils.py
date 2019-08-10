import numpy as np
from typing import List
from hw1_knn import KNN

def get_entropy(branch):
    entropy = 0
    denom = sum(branch)
    for subclass in branch:
        prob = subclass / denom
        if subclass == 0:
            entropy += 0
        else:
            entropy -= (prob * np.log2(prob))
    return entropy


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    #raise NotImplementedError

    entropy_list = []
    denom_list = []
    total_elems = 0  

    for branch in branches:
        entropy = 0
        denom = sum(branch)
        total_elems += denom
        for subclass in branch:
            prob = subclass / denom
            if subclass == 0:
                entropy += 0
            else:
                entropy -= (prob * np.log2(prob))
        entropy_list.append(entropy)
        denom_list.append(denom)

    ig = S - sum([a * b for a, b in zip(denom_list, entropy_list)]) / total_elems
    return ig


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List[any]
    #raise NotImplementedError
    
    y_pred = decisionTree.predict(X_test)
    best_acc = get_accuracy(y_pred, y_test)
    #print("Accuracy: " + str(best_acc))
    
    node = decisionTree.root_node
    best_node = node

    #for c in node.children:
    #    print(c)
    #    new_node, best_acc, decisionTree = reduced_error_prunning_helper(decisionTree, c, X_test, y_test, best_acc, best_node)

    #if new_node == best_node:
    #    print("No better node found, returning")
    #    return
    #else:
    #    reduced_error_prunning(decisionTree, X_test, y_test)

    """
    while ret_node != best_node:
        ret_node, ret_acc = reduced_error_prunning_helper(decisionTree, node, X_test, y_test, best_node, best_acc)
        ret_node.children = []
        ret_node.dim_split = None
        ret_node.splittable = False
        print(ret_node)
        print(best_node)
        best_acc = ret_acc
    """
    acc, newNode = reduced_error_prunning_helper(decisionTree, node, X_test, y_test, best_acc, best_node)

#def reduced_error_prunning_helper(decisionTree, node, X_test, y_test, best_acc, best_node):
def reduced_error_prunning_helper(decisionTree, node, X_test, y_test, best_acc, best_node):
    if node.splittable == False:
        #print("Reached leaf, returning")
        #print()
        return best_acc, best_node

    #print("Current features: " + str(node.features))
    new_y_pred = decisionTree.predict(X_test)
    new_acc = get_accuracy(new_y_pred, y_test)
    #print("Current accuracy: " + str(new_acc))

    temp_children = []
    temp_children.extend(node.children)
    temp_dim_split = node.dim_split

    #print("Pruning node")
    node.children = []
    node.dim_split = None
    node.splittable = False

    new_y_pred = decisionTree.predict(X_test)
    new_acc = get_accuracy(new_y_pred, y_test)
    #print("New accuracy: " + str(new_acc))

    #print("Restoring node")
    node.children = temp_children
    node.dim_split = temp_dim_split
    node.splittable = True

    if new_acc > best_acc:
        #print("New best node: " + str(node.features))
        best_acc = new_acc
        #print("new best acc " + str(best_acc))
        best_node = node

    for idc, child in enumerate(node.children):
        #print("Child number " + str(idc) + " with features: " + str(child.features))
        best_acc, best_node = reduced_error_prunning_helper(decisionTree, child, X_test, y_test, best_acc, best_node)

        return best_acc, best_node
    #print("--------")

    #print("Best node: " + str(best_node.features))
    """
    if node.splittable == False:
        print("Found a leaf, returning")
        return node, best_acc, decisionTree 

    temp_children = []
    temp_children.extend(node.children)
    temp_dim_split = node.dim_split

    print("Pruning node")
    node.children = []
    node.dim_split = None
    node.splittable = False

    new_y_pred = decisionTree.predict(X_test)
    new_acc = get_accuracy(new_y_pred, y_test)
    print("New accuracy: " + str(new_acc))

    if new_acc <= best_acc:
        print("New accuracy no better")
        node.children = temp_children
        node.dim_split = temp_dim_split
        node.splittable = True
    elif new_acc > best_acc:
        print("Found better accuracy")
        best_acc = new_acc
        best_node = node
        print("Restoring previous vals")
        node.children = temp_children
        node.dim_split = temp_dim_split
        node.splittable = True
        return best_node, best_acc, decisionTree

    for child in node.children:
        print("recursing")
        best_node, best_acc, decisionTree = reduced_error_prunning_helper(decisionTree, child, X_test, y_test, best_acc, best_node)

    print("reached end, returning")
    return best_node, best_acc, decisionTree
    """
    
        

def get_accuracy(pred, actual):
    #assert (len(pred) == len(actual))
    accurate = 0
    for a, b in zip(pred, actual):
        if a == b:
            accurate += 1
    return accurate / len(pred)

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    #raise NotImplementedError
    s = 0
    for i, j in zip(real_labels, predicted_labels):
        s += i * j
    
    s *= 2
    denom = sum(real_labels) + sum(predicted_labels)

    if denom == 0:
        return 0
    else:
        return s / denom

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    dist = 0
    for p, q in zip(point1, point2):
        dist += np.square(p - q)
    
    ret = np.sqrt(dist)
    return ret


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    dist = 0
    for p, q in zip(point1, point2):
        dist += p * q
    
    return dist


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    dist = 0
    for p, q in zip(point1, point2):
        dist += np.square(p - q)
    
    prod = (-0.5) * dist
    ret = np.exp(prod) * -1
    return ret


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    
    len1 = 0
    len2 = 0
    numer = 0
    for p, q in zip(point1, point2):
        numer += p * q
        len1 += np.square(p)
        len2 += np.square(q)

    denom = np.sqrt(len1) * np.sqrt(len2)

    if denom == 0:
        return 1

    return 1 - (numer / denom)


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    #raise NotImplementedError
    best_model = None
    best_k = 0
    best_func = distance_funcs['euclidean']
    max_score = 0
    n = 30

    if len(Xtrain) < 30:
        n = len(Xtrain) - 1  

    for func in distance_funcs:
        for k in range(1, n, 2):
            model = KNN(k, distance_funcs[func])
            model.train(Xtrain, ytrain)
            predicted = model.predict(Xval)
            
            temp_f1 = f1_score(yval, predicted)
            
            print('[part 1.1] {name}\tk: {k:d}\t'.format(name = func, k = k) +
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score = temp_f1))
            print()
            
            if temp_f1 > max_score:
                max_score = temp_f1
                best_model = model
                best_k = k
                best_func = func
                              
            if temp_f1 == max_score:
                if k < best_k:
                    max_score = temp_f1
                    best_model = model
                    best_k = k
                    best_func = func

    
    
    print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name = best_func, best_k = best_k) +
        'test f1 score: {test_f1_score:.5f}'.format(test_f1_score = max_score))
    print()
          
    """
    #Dont change any print statement

    
    print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=k) +
    'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
    'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    print()
    
    print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=name, best_k=best_k) +
    'test f1 score: {test_f1_score:.5f}'.format(test_f1_score=test_f1_score))
    print()
    
    """
        
    return best_model, best_k, best_func



# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    #raise NotImplementedError

    best_model = None
    best_k = 0
    best_func = distance_funcs['euclidean']
    best_scaler = scaling_classes['min_max_scale']
    max_score = 0
    n = 30

    if len(Xtrain) < 30:
        n = len(Xtrain) - 1
        
    for sc in scaling_classes:
        scaler = scaling_classes[sc]()
        
        scaled_train = scaler(Xtrain)
        scaled_val = scaler(Xval)
        
        for func in distance_funcs:
            for k in range(1, n, 2):
                model = KNN(k, distance_funcs[func])
                model.train(scaled_train, ytrain)
                predicted = model.predict(scaled_val)
                
                temp_f1 = f1_score(yval, predicted)
                #valid_f1_score = f1_score(yval, scaled_val)
                """
                print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name = func, scaling_name = sc, k = k) +
                      'train: {train_f1_score:.5f}\t'.format(train_f1_score = temp_f1) +
                      'valid: {valid_f1_score:.5f}'.format(valid_f1_score = valid_f1_score))
                print()
                """
                
                print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name = func, scaling_name = sc, k = k) +
                      'train: {train_f1_score:.5f}\t'.format(train_f1_score = temp_f1))
                print()
                
                if temp_f1 > max_score:
                    max_score = temp_f1
                    best_model = model
                    best_k = k
                    best_func = func
                    best_scaler = sc
                                      
                if temp_f1 == max_score:
                    if k < best_k:
                        max_score = temp_f1
                        best_model = model
                        best_k = k
                        best_func = func
                        best_scaler = sc
    
    print('[part 1.2] {name}\t{scaling_name}\t'.format(name = best_func, scaling_name = best_scaler) +
          'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k = best_k, test_f1_score = max_score))
    print()
    

    """
    #Dont change any print statement
    print('[part 1.2] {name}\t{scaling_name}\tk: {k:d}\t'.format(name=name, scaling_name=scaling_name, k=k) +
    'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
    'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
    
    print()
    print('[part 1.2] {name}\t{scaling_name}\t'.format(name=name, scaling_name=scaling_name) +
    'best_k: {best_k:d}\ttest: {test_f1_score:.5f}'.format(best_k=best_k, test_f1_score=test_f1_score))
    print()
    """
            
    return best_model, best_k, best_func, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        #raise NotImplementedError
        ret = []
        for f in features:
            denom = 0
            for x in f:
                denom += np.square(x)
            denom = np.sqrt(denom)
            if denom == 0:
                ret.append([0 for x in f])
            else:
                temp = [i / denom for i in f]
                ret.append(temp)
        
        return ret


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.min = None
        self.max = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        #raise NotImplementedError

        ret = []

        if self.min == None:
            min_list = []
            max_list = []

            for i in range(len(features[0])):
                pos_vals = [elem[i] for elem in features]
                min_val = min(pos_vals)
                max_val = max(pos_vals)
                min_list.append(min_val)
                max_list.append(max_val)

            self.min = min_list
            self.max = max_list

        for f in features:
            elem = []
            for i in range(len(f)):
                if self.max[i] - self.min[i] == 0:
                    elem.append(0)
                else:
                    elem.append((f[i] - self.min[i]) / (self.max[i] - self.min[i]))
            ret.append(elem)

        return ret
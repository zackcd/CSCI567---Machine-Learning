import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            #print("PREDICTING LABEL FOR: " + str(feature))
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
            #print("PREDICTED VAL: " + str(pred))
            #print("-------------------")
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        #raise NotImplementedError
        #print("self.features:")
        #print(self.features)
        all_same = all(x == self.labels[0] for x in self.labels)
        all_empty = all(x == [] for x in self.features)
        #if Examples have the same class, return a leaf with this class
        if all_same == True:
            #newNode = TreeNode(self.features, [self.labels[0]], 1)
            #newNode = TreeNode(self.features, self.labels, 1)
            newNode = TreeNode([], self.labels, 1)
            newNode.splittable = False
            self.children.append(newNode)
            #print("All examples same, leaf created")
            return
        #elif empty_features:
        elif self.features == [] or all_empty == True:
            #newNode = TreeNode(self.features, self.labels, 1)
            newNode = TreeNode(self.features, [self.cls_max], 1)
            newNode.splittable = False
            self.children.append(newNode)
            #print("All features empty, leaf created")
            return
        else:
            #Find attribute to split on:
            A, split_features = self.get_split_attribute()

            self.dim_split = A
            self.feature_uniq_split = list(np.unique(split_features))

            #print("Unique features")
            #print(self.feature_uniq_split)

            for f in self.feature_uniq_split:
                #print("f: " + str(f))
                remaining_features = []
                remaining_labels = []
                for ida, a in enumerate(self.features):
                    if a[A] == f:
                        temp_feature = a[0:A] + a[A + 1:]
                        if len(temp_feature) > 0:
                            remaining_features.append(temp_feature)
                        remaining_labels.append(self.labels[ida])
                remaining_cls = len(np.unique(remaining_labels))

                #print("Remaining features:")
                #print(remaining_features)
                #print("Remaining labels:")
                #print(remaining_labels)

                if remaining_features == []:
                    max_class = 0
                    count_max = 0
                    for label in np.unique(remaining_labels):
                        if remaining_labels.count(label) > count_max:
                            count_max = remaining_labels.count(label)
                            max_class = label
                    newNode = TreeNode(remaining_features, [max_class], 1)
                    newNode.splittable = False
                    self.children.append(newNode)
                    #print("All examples same, creating leaf node")
                else:
                    newNode = TreeNode(remaining_features, remaining_labels, remaining_cls)
                    #newNode = TreeNode(sorted_features, sorted_labels, remaining_cls)
                    self.children.append(newNode)
                    newNode.split()
                #print()
            #print("-----------")
        return

    def get_split_attribute(self):
        ent = [self.labels.count(x) for x in set(self.labels)]
        S = Util.get_entropy(ent)
        A = 0
        max_ig = 0
        split_features = []
        for a in range(len(self.features[0])):
            newFeatures = []
            branches = {}
            for idf, f in enumerate(self.features):
                newFeatures.append(f[a])
                if f[a] in branches:
                    branches[f[a]].append(self.labels[idf])
                else:
                    branches[f[a]] = [self.labels[idf]]

            counts = []
            for b in branches:
                counts.append([branches[b].count(x) for x in set(branches[b])])
            ig = Util.Information_Gain(S, counts)
            if ig == max_ig:
                if len(newFeatures) > len(split_features):
                    max_ig = ig
                    A = a
                    split_features = newFeatures
            if ig > max_ig:
                max_ig = ig
                A = a
                split_features = newFeatures
        #print("Attribute to split on: " + str(A))
        return A, split_features


    # TODO: predict the branch or the class
    def predict(self, feature):
        #print("Self.features: " + str(self.features))
        # feature: List[any]
        # return: int
        #raise NotImplementedError
        #print(self.features)
        a = self.dim_split
        if a == None:
            #print("Reached a leaf, returning prediction: " + str(self.cls_max))
            return self.cls_max
        if feature == []:
            #print("Examples empty, returning parent's max_cls")
            return self.cls_max
        #print(feature[a])
        noMatch = False
        unique_features = np.unique([f[a] for f in self.features])
        feature_mapping = {k: v for v, k in enumerate(unique_features)}
        #print(feature_mapping)
        for idf, f in enumerate(self.features):
            #print("ifd: " + str(idf))
            #print("current feature: " + str(f))
            #print("element " + str(a) + " in feature " + str(feature) + " is " + str(feature[a]))
            #print("f[a]: " + str(f[a]))
            noMatch = True
            if feature[a] == f[a]:
                #print("Match")
                newFeature = []
                newFeature.extend(feature)
                del newFeature[a]
                #print("Remaining attributes in feature: " + str(newFeature))

                return self.children[feature_mapping[f[a]]].predict(newFeature)
        if noMatch == True:
            #print("No match: Examples empty, returning parent's max_cls")
            return self.cls_max
        #print()

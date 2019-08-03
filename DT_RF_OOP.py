import pandas as pd
import numpy as np
import math
from collections import Counter

class TreeNode(object):
    #A node class for a decision tree.
    def __init__(self):
        self.column = None  #(int) index of feature to split on
        self.value = None  #value of the feature to split on
        self.categorical = True  # (bool) whether or not node is split on categorial feature
        self.name = None    # (string) name of feature (or name of class in the case of a list)
        self.left = None    # (TreeNode) left child
        self.right = None   # (TreeNode) right child
        self.leaf = False   # (bool)   true if node is a leaf, false otherwise
        self.classes = Counter()  # only necessary for leaf node:key is class name and value is count of the count of data points
                                  # that terminate at this leaf

    def predict_one(self, x):
         # x: 1d numpy array (single data point)
         # Return y: predicted label for a single data point.
        if self.leaf:
            return(self.name)
        
        col_value = x[self.column]

        if self.categorical:
            if col_value == self.value:
                return(self.left.predict_one(x))
            else:
                return(self.right.predict_one(x))
        else:
            if col_value < self.value:
                return(self.left.predict_one(x))
            else:
                return(self.right.predict_one(x))

######################################################################################
class DecisionTree(object):
    #A decision tree class.
    def __init__(self, impurity_criterion='entropy', num_features=None):

        self.root = None  # root Node
        self.feature_names = None  # string names of features (for interpreting
                                   # the tree)
        self.categorical = None  # Boolean array of whether variable is
                                 # categorical (or continuous)
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini
        self.num_features = num_features

    def fit(self, X, y, feature_names=None):
        # X: 2d numpy array of features in column and data point as rows
        # y: 1d numpy array
        # feature_names: numpy array of strings
        
        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        is_categorical = lambda x: isinstance(x, str) or \
                                   isinstance(x, bool)
        self.categorical = np.vectorize(is_categorical)(X[0])

        if not self.num_features:
            self.num_features = X.shape[1]

        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        # Recursively build the decision tree. Return the root node.
        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y)

        if index is None or len(np.unique(y)) == 1:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)
        return(node)

    def _entropy(self, y):
        # Return the entropy of the array y.
        
        n = y.shape[0]
        summation = 0
        for c_i in np.unique(y):
            prob = np.mean(y == c_i)
            summation += prob * np.log2(prob)
        return -summation

    def _gini(self, y):
        # Return the gini impurity of the array y.

        n = y.shape[0]
        summation = 0
        for c_i in np.unique(y):
            prob = np.mean(y == c_i)
            summation += prob**2
        return(1 - summation)

    def _make_split(self, X, y, split_index, split_value):
        # Return the two subsets of the dataset achieved by the given feature and value to split on.
        split_col = X[:, split_index]
        if self.categorical[split_index]:
            idx = (split_col == split_value)
        else:
            idx = (split_col < split_value)
        return X[idx], y[idx], X[~idx], y[~idx]

    def _information_gain(self, y, y1, y2):

        # Return the information gain of making the given split.
        # Use self.impurity_criterion(y) rather than calling _entropy or _gini directly.

        n = y.shape[0]
        weighted_child_imp = 0
        for y_i in (y1, y2):
            weighted_child_imp += self.impurity_criterion(y_i) * y_i.shape[0] / n
        return(self.impurity_criterion(y) - weighted_child_imp)

    def _choose_split_index(self, X, y):
        # Returns:
        # index: int (index of feature)
        # value: int/float/bool/str (value of feature)
        #  splits: (2d array, 1d array, 2d array, 1d array)
        # Determine which feature and value to split on. Return the index and
        #value of the optimal split along with the split of the dataset.
        split_index, split_value, splits = None, None, None
        feature_indices = np.random.choice(X.shape[1], self.num_features, replace=False)
        max_gain = 0

        for i in feature_indices:
            values = np.unique(X[:, i])
            if len(values) < 2:
                continue
            for val in values:
                temp_splits = self._make_split(X, y, i, val)
                X1, y1, X2, y2 = temp_splits
                gain = self._information_gain(y, y1, y2)
                if gain > max_gain:
                    max_gain = gain
                    split_index, split_value = i, val
                    splits = temp_splits
        return split_index, split_value, splits

    def predict(self, X):
        return(np.array([self.root.predict_one(row) for row in X]))


######################################################################################

class RandomForest(object):

    #Random Forest Class
    def __init__(self,num_trees,num_features):

        #num_trees: number of trees to create in the forest
        #num_features: number of features to consider when choosing the best split for each node of the decision trees        
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None
        
    def fit(self, X, y):
        #X: a 2d numpy array of feature matrix of training data set
        #y: a numpy array of labels for training data set
        self.forest = self.build_forest(X, y, self.num_trees, self.num_features)

    def build_forest(self, X, y, num_trees, num_features):
        #List of DecisionTrees    
    
        forest = []
        for i in range(num_trees):
            spl_ind = np.random.choice(X.shape[0], X.shape[0], replace=True)
            spl_X = np.array(X[spl_ind])
            spl_y = np.array(y[spl_ind])
            dec_tree = DecisionTree(num_features=self.num_features)
            dec_tree.fit(spl_X, spl_y)
            forest.append(dec_tree)
        return(forest)

    def predict(self, X):

        #Return a numpy array of the labels predicted for the given test data.
        answers = np.array([tree.predict(X) for tree in self.forest]).T
        return(np.array([Counter(row).most_common(1)[0][0] for row in answers]))

    def score(self, X, y):
        #Return the accuracy of the Random Forest for the given test data and labels.
        return ((self.predict(X) == y).mean())


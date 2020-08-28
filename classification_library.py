import numpy as np
import pandas as pd
from utils import rgb2gray, get_regression_data, visualise_regression_data, colors, show_data
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
import sklearn.datasets
import sklearn.tree
from sklearn.metrics import accuracy_score
import sys
sys.path.append('..') # add utils file to path
import json


def create_bootstrapped_dataset(existing_X, existing_Y, size):
    """Create a single bootstrapped dataset"""
    idxs = np.random.choice(np.arange(len(existing_X)), size=size, replace=True) # randomly sample indices with replacement
    return existing_X[idxs], existing_Y[idxs] # return examples at these indices

def project_into_subspace(X, feature_idxs):
    """
    Returns only the features of a dataset X at the indices provided 
    feature_idxs should be a list of integers representing the indices of the features that should remain 
    """
    return X[:, feature_idxs]

class RandomForest:
    def __init__(self, n_trees=10, max_depth=4, max_samples=10):
        self.n_trees = n_trees # how many trees in the forest
        self.max_depth = max_depth # what is the max depth of each tree
        self.trees = [] # init an empty list of trees
        self.max_samples = max_samples # how many samples from the whole dataset should each tree be trained on

    def fit(self, X, Y):
        """Fits a bunch of decision trees to input X and output Y"""
        for tree_idx in range(self.n_trees): # for each bootstrapped dataset
            bootstrapped_X, bootstrapped_Y = create_bootstrapped_dataset(X, Y, size=self.max_samples) # get features and labels of new bootstrapped dataset
            n_features = np.random.choice(range(1, bootstrapped_X.shape[1])) # choose how many features this tree will use to make predictions
            subspace_feature_indices = np.random.choice(range(bootstrapped_X.shape[1]), size=n_features) # randomly choose that many features to use as inputs
            projected_X = project_into_subspace(bootstrapped_X, subspace_feature_indices) # remove unused features from the dataset
            tree = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth) # init a decision tree
            tree.fit(projected_X, bootstrapped_Y) # fit the tree on these examples
            tree.feature_indices = subspace_feature_indices # give the tree a new attribute: which features were used 
            self.trees.append(tree) # add this tree to the list of trees

    def predict(self, X):
        """Uses the fitted decision trees to return predictions"""
        predictions = np.zeros((len(X), self.n_trees)) # empty array of prediction with shape n_examples x n_trees
        for tree_idx, tree in enumerate(self.trees): # for each tree in our forest
            x = project_into_subspace(X, tree.feature_indices) # throw away some features of each input example for this tree to predict based on those alone
            predictions[:, tree_idx] = tree.predict(x) # predict integer label
        prediction = np.mean(predictions, axis=1) # average predictions from different models
        # prediction = np.round(prediction).astype(int) # comment this line to show probability confidences of predictions rather than integer predictions
        return prediction

    def __repr__(self):
        """Returns a string representation of the random forest"""
        forest = [] # init empty list of trees
        for idx, tree in enumerate(self.trees): # for each tree in the forest
            forest.append({ # add a dictionary of info about the tree
                'depth': tree.max_depth, # how many binary splits?
                'features': tree.feature_indices.tolist() # which features is it using
            })
        return json.dumps(forest, indent=4) # convert dict to string with nice indentation


class AdaBoost:
    def __init__(self, n_layers=20):
        self.n_layers = n_layers
        self.models = [] # init empty list of models

    def sample(self, X, Y, weights):
        idxs = np.random.choice(range(len(X)), size=len(X), replace=True, p=weights)
        X = X[idxs]
        Y = Y[idxs]
        return X, Y

    def calc_model_error(self, predictions, labels, example_weights):
        """Compute classifier error rate"""
        diff = predictions == labels
        diff = diff.astype(float)
        diff *= example_weights
        # weighted_diffs = weights * diff
        return np.mean(diff)

    def encode_labels(self, labels):
        labels[labels == 0] = -1
        labels[labels == 1] = +1
        return labels

    def calc_model_weight(self, error, delta=0.01):
        z = (1 - error) / (error + delta) + delta
        return 0.5 * np.log(z)

    def update_weights(self, predictions, labels, model_weight):
        weights = np.exp(- model_weight * predictions * labels)
        weights /= np.sum(weights)
        return weights

    def fit(self, X, Y):
        labels = self.encode_labels(Y)
        example_weights = np.full(len(X), 1/len(X)) # assign initial importance of classifying each example as uniform and equal
        for layer_idx in range(self.n_layers):
            model = sklearn.tree.DecisionTreeClassifier(max_depth=1)
            bootstrapped_X, bootstrapped_Y = self.sample(X, Y, example_weights)
            model.fit(bootstrapped_X, bootstrapped_Y)
            predictions = model.predict(X) # make predictions for all examples
            model_error = self.calc_model_error(predictions, labels, example_weights)
            model_weight = self.calc_model_weight(model_error)
            model.weight = model_weight
            self.models.append(model)
            example_weights = self.update_weights(predictions, labels, model_weight)
            # print(f'trained model {layer_idx}')
            # print()

    def predict(self, X):
        prediction = np.zeros(len(X))
        for model in self.models:
            prediction += model.weight * model.predict(X)
        return np.sign(prediction)

    def __repr__(self):
        return json.dumps([m.weight for m in self.models])
        return json.dumps([
            {
                'weight': model.weight
            }
            for model in self.models
        ], indent=4)
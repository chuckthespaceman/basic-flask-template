import pandas as pd
import pickle

from sklearn.ensemble import BaggingClassifier
import numpy as np

# Custom implementation of PART (partial decision tree)
import numpy as np

class Node:
    def _init_(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold for binary split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Predicted value for leaf nodes

class PART:
    def _init_(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self.predict(inputs, self.tree) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or len(np.unique(y)) == 1 or len(y) < self.min_samples_leaf:
            return Node(value=self._most_common_label(y))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # Split the data
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features_):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                gini = self._gini_impurity(y[left_indices], y[~left_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, left_labels, right_labels):
        p_left = len(left_labels) / (len(left_labels) + len(right_labels))
        p_right = len(right_labels) / (len(left_labels) + len(right_labels))

        gini_left = 1 - sum([(np.sum(left_labels == c) / len(left_labels)) ** 2 for c in np.unique(left_labels)])
        gini_right = 1 - sum([(np.sum(right_labels == c) / len(right_labels)) ** 2 for c in np.unique(right_labels)])

        gini = p_left * gini_left + p_right * gini_right
        return gini

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def _predict(self, inputs, tree):
        if tree.value is not None:
            return tree.value

        if inputs[tree.feature] < tree.threshold:
            return self._predict(inputs, tree.left)
        else:
            return self._predict(inputs, tree.right)

    def get_params(self, deep=True):
        return {'max_depth': self.max_depth, 'min_samples_leaf': self.min_samples_leaf}

    def set_params(self, **params):
        if 'max_depth' in params:
            self.max_depth = params['max_depth']
        if 'min_samples_leaf' in params:
            self.min_samples_leaf = params['min_samples_leaf']
        return self

# Load the model from the pickle file
with open('basic-flask-template\\templates\\PART_model.pkl', 'rb') as file:
  loaded_model = pickle.load(file)

print("Model loaded from './Crop_recommendation.csv'")

N = input("Enter the value of N: ")
P = input("Enter the value of P: ")
K = input("Enter the value of K: ")
temperature = input("Enter the value of temperature: ")
humidity = input("Enter the value of humidity: ")
ph = input("Enter the value of ph: ")
rainfall = input("Enter the value of rainfall: ")

data = {
    'N': [float(N)],
    'P': [float(P)],
    'K': [float(K)],
    'temperature': [float(temperature)],
    'humidity': [float(humidity)],
    'ph': [float(ph)],
    'rainfall': [float(rainfall)],
}
df = pd.DataFrame(data)

# Use the loaded model to make predictions
predictions = loaded_model.predict(df)
print("Predictions:", predictions)


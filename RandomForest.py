import numpy as np
from collections import Counter
from typing import List, Tuple

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # Tree structure
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X: np.ndarray, y: np.ndarray, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            self.value = Counter(y).most_common(1)[0][0]
            return

        # Find the best split
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature_idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no good split is found, make this a leaf node
        if best_gain == -1:
            self.value = Counter(y).most_common(1)[0][0]
            return

        # Create child nodes
        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.left = DecisionTree(self.max_depth, self.min_samples_split)
        self.right = DecisionTree(self.max_depth, self.min_samples_split)

        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right.fit(X[right_mask], y[right_mask], depth + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray):
        # If this is a leaf node, return its value
        if self.value is not None:
            return self.value
        
        # Otherwise, traverse left or right based on the split
        if x[self.feature_idx] < self.threshold:
            return self.left._predict_single(x)
        return self.right._predict_single(x)

    def _information_gain(self, y: np.ndarray, feature: np.ndarray, threshold: float) -> float:
        # Calculate entropy of parent node
        parent_entropy = self._entropy(y)

        # Create masks for left and right splits
        left_mask = feature < threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        # Calculate weighted entropy of children
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = n - n_left

        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])
        weighted_entropy = (n_left/n) * left_entropy + (n_right/n) * right_entropy

        # Return information gain
        return parent_entropy - weighted_entropy

    def _entropy(self, y: np.ndarray) -> float:
        # Calculate entropy of a node
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Create and train a new decision tree
            tree = DecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Take majority vote for each sample
        return np.array([Counter(predictions).most_common(1)[0][0] 
                        for predictions in tree_predictions.T])

# Let's test our implementation with a simple example
if __name__ == "__main__":
    # Create a simple dataset
    X = np.array([
        [1, 2], [2, 3], [3, 4], [4, 5],  # Class 0
        [5, 6], [6, 7], [7, 8], [8, 9]   # Class 1
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Create and train the random forest
    rf = RandomForest(n_trees=5, max_depth=3)
    rf.fit(X, y)

    # Make predictions
    predictions = rf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy:.2f}")

    # Test on new data
    X_test = np.array([[2.5, 3.5], [6.5, 7.5]])
    predictions = rf.predict(X_test)
    print(f"Predictions for new data: {predictions}")
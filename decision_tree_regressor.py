import copy
import numpy as np
import eval


# Calculates the mean square error for a particular set of data coming in to the node
def calc_error(df, label_column):
    if len(df) > 0:
        r_mean = df[label_column].mean()
        error = 0
        for i in range(len(df)):
            error += np.power(df[label_column].iloc[i] - r_mean, 2)
        return error / len(df)
    else:
        return 0


# Finds the feature with the lowest mean square error
def find_lowest_error_feature(df, label_column):
    lowest_error = np.inf
    lowest_error_feature = None
    lowest_error_splits = []
    lowest_error_uniques = []

    for feature in df.loc[:, df.columns != label_column].columns:

        # For categorical data
        if df[feature].dtype in DecisionTreeRegressor.dtype_class:

            # Split the dataset by class
            df_splits = []
            uniques = df[feature].unique()
            for unique in uniques:
                df_splits += [df.loc[df[feature] == unique]]

            # Find the combined error of the splits
            splits_error = 0
            for df_split in df_splits:
                splits_error += calc_error(df_split, label_column)

            # If the error is lower than the current min error replace
            if splits_error < lowest_error:
                lowest_error = splits_error
                lowest_error_feature = feature
                lowest_error_splits = df_splits
                lowest_error_uniques = uniques

        # For numerical data
        else:
            # Find the midpoint in the feature range
            midpoint = df[feature].min() + ((df[feature].max() - df[feature].min()) / 2)

            # Split the dataset on the midpoint
            df_splits = []
            df_splits += [df.loc[df[feature] < midpoint]]
            df_splits += [df.loc[df[feature] >= midpoint]]

            # Find the combined error of the splits
            splits_error = 0
            for df_split in df_splits:
                splits_error += calc_error(df_split, label_column)

            # If the error is lower than the current min error replace
            if splits_error < lowest_error:
                lowest_error = splits_error
                lowest_error_feature = feature
                lowest_error_splits = df_splits

    return lowest_error_feature, lowest_error_splits, lowest_error_uniques, lowest_error


# Prints the contents of a node and the node's children
def print_tree(node, parent_node, branch):
    if parent_node is not None:
        print(f"{parent_node.data} branch={branch}, child node={node.data}")
    else:
        print(f"top node={node.data}")

    for child in node.children.keys():
        print_tree(node.children[child], node, child)


# Decision Tree Classifier class
# Contains the top node of the tree and the theta
class DecisionTreeRegressor:
    top_node = None
    theta = 0
    dtype_class = ['object', 'bool']
    mean_or_median = 'mean'

    def __init__(self, theta=0, mean_or_median='mean'):
        self.theta = theta
        self.mean_or_median = mean_or_median

    # Builds a classification tree
    def generate_tree(self, df, label_column, node):

        # First calculate the error for the node
        node_error = calc_error(df, label_column)

        # Now find the feature with the lowest error
        lowest_error_feature, lowest_error_splits, lowest_error_uniques, lowest_error = find_lowest_error_feature(df, label_column)
        node.data = lowest_error_feature
        node.df = df

        # If the error is below the threshold create a leaf
        if node_error <= self.theta or lowest_error_feature is None:
            if self.mean_or_median == 'mean':
                leaf_value = df[label_column].mean()
            else:
                leaf_value = df[label_column].median()
            node.data = leaf_value
            node.df = None
            return

        # Now create children
        if df[lowest_error_feature].dtype in DecisionTreeRegressor.dtype_class:
            node.type = 'categorical'

            # For each unique-split pair create a child and generate for this split dataframe
            for i in range(len(lowest_error_splits)):
                child_node = DecisionTreeRegNode(None, None)
                node.children[lowest_error_uniques[i]] = child_node
                self.generate_tree(lowest_error_splits[i].drop(columns=[lowest_error_feature]), label_column, child_node)

            # Finally create a default node with the average value for categories not trained on
            if self.mean_or_median == 'mean':
                child_node = DecisionTreeRegNode(df[label_column].mean(), None)
            else:
                child_node = DecisionTreeRegNode(df[label_column].median(), None)
            node.children['default'] = child_node

        else:
            node.type = 'numerical'

            # Create 1st child for items < the midpoint
            child_node = DecisionTreeRegNode(None, None)
            midpoint = df[lowest_error_feature].min() + ((df[lowest_error_feature].max() - df[lowest_error_feature].min()) / 2)
            node.children[midpoint] = child_node
            self.generate_tree(lowest_error_splits[0].drop(columns=[lowest_error_feature]), label_column, child_node)

            # Create 2nd child for remaining items >= the midpoint
            child_node = DecisionTreeRegNode(None, None)
            node.children['upper'] = child_node
            self.generate_tree(lowest_error_splits[1].drop(columns=[lowest_error_feature]), label_column, child_node)

    # Traverses the tree given a particular row of data
    def traverse_tree(self, df_row, node, verbose=False):
        if verbose:
            print(f"Searching node {node.data}")

        if len(node.children) > 0:
            child = None
            feature_value = df_row[node.data].values[0]

            # Categorical if the feature is present in the tree, go to it
            # otherwise take the 'default' branch (a leaf with the most common class value)
            if node.type == "categorical":
                if verbose:
                    print(f"Taking branch {feature_value}")
                if feature_value in node.children.keys():
                    child = node.children[feature_value]
                else:
                    child = node.children['default']

            # Numerical use the closest child that is greater than the feature value
            # if the feature value is higher than all the children use the 'upper' child
            if node.type == 'numerical':
                midpoints = list(node.children.keys())
                midpoints.remove("upper")
                for midpoint in midpoints:
                    if feature_value < midpoint:
                        child = node.children[midpoint]
                        break
                if child is None:
                    if verbose:
                        print(f"Taking branch for largest values")
                    child = node.children['upper']

            return self.traverse_tree(df_row, child, verbose=verbose)
        else:
            if verbose:
                print(f"Node is a leaf, returning {node.data}")
            return node.data

    # Generates a tree and makes predictions for the test data
    def fit_predict_regression(self, df_trn, df_tst, label_column):
        self.top_node = DecisionTreeRegNode(None, None)
        self.generate_tree(df_trn, label_column, self.top_node)

        y = []
        for i in range(len(df_tst)):
            df_row = df_tst.iloc[i]
            y += [self.traverse_tree(df_tst, self.top_node)]
        return y

    # Makes predictions for an existing tree
    def predict_regression(self, df_tst):
        y = []
        for i in range(len(df_tst)):
            df_row = df_tst.iloc[i]
            y += [self.traverse_tree(df_tst, self.top_node)]
        return y

    # Returns a list of nodes in the tree with children (subtrees)
    def find_subtrees(self, node, subtrees):
        if len(node.children) > 0:
            subtrees += [node]
            for key in node.children.keys():
                if key != 'default' and key != 'upper':
                    self.find_subtrees(node.children[key], subtrees)

    # Prunes the list of subtrees
    def prune_subtrees(self, subtrees, label_column, df_prune):
        prune_count = 0
        y_truth = df_prune[label_column].values.ravel()

        for subtree in subtrees:
            # First find the accuracy of the old tree
            y_pred = self.predict_regression(df_prune)
            mse_pred = eval.eval_mse(y_truth, y_pred)

            # Replace the node with a leaf containing mean of the dataset at the leaf
            old_node = copy.deepcopy(subtree)
            subtree.children = {}
            subtree.type = None
            subtree.df = None
            subtree.data = old_node.df[label_column].mean()

            # Now find the accuracy on the smaller tree
            y_pruned = self.predict_regression(df_prune)
            mse_pruned = eval.eval_mse(y_truth, y_pruned)

            # only if the pruned mse > the original mse, bring back the old node
            if mse_pruned > mse_pred:
                subtree.children = old_node.children
                subtree.type = old_node.type
                subtree.df = old_node.df
                subtree.data = old_node.data
            else:
                prune_count += 1


# Decision Tree Node
# Contains the data (a feature)
# children - a dictionary with feature values to nodes
# df - the dataset for this node
# type - whether the node's feature is a categorical or numerical value
class DecisionTreeRegNode:
    def __init__(self, data, df):
        self.data = data
        self.df = df
        self.type = None
        self.children = {}

    def find_child(self, child_data):
        return self.children[child_data]

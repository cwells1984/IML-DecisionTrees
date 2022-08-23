import copy
import eval
import numpy as np


# Calculate the entropy for the dataset in the node
def calc_entropy(df, label_column):
    entropy_sum = 0
    for target_class in df[label_column].unique():
        p = df[label_column].value_counts()[target_class] / len(df)
        entropy_sum += -1 * p * np.log2(p)
    return entropy_sum


# Calculate the feature entropy for a particular feature in the node's dataset
def calc_feature_entropy(df, feature, label_column):
    feature_entropy_sum = 0

    for feature_class in df[feature].unique():
        p = df[feature].value_counts()[feature_class] / len(df)
        entropy_sum = 0
        for target_class in df[label_column].unique():
            q = len(df.loc[df[feature] == feature_class].loc[df[label_column] == target_class]) / \
                len(df.loc[df[feature] == feature_class])
            if q != 0:
                entropy_sum += -1 * q * np.log2(q)

        feature_entropy_sum += p * entropy_sum

    return feature_entropy_sum


# Finds the feature with the highest gain in the node's dataset
def find_highest_gain(df, label_column, node_entropy):
    highest_gain = 0
    highest_gain_feature = None

    for feature in df.loc[:, df.columns != label_column].columns:
        feature_entropy = calc_feature_entropy(df, feature, label_column)
        gain = node_entropy - feature_entropy
        if gain > highest_gain:
            highest_gain = gain
            highest_gain_feature = feature

    return highest_gain_feature, highest_gain


# Returns a list of midpoints for a particular feature at class boundaries
def find_midpoints(df, feature, target):
    # Creates an empty set containing the midpoints
    midpoints = set()

    # Sort by the feature
    df_sorted = df.sort_values(by=[feature])

    # Now for each row, see if the class has changed. If it has, find the midpoint and add to the set
    # If both features are the same the midpoint will be 0 and should be ignored
    for i in range(len(df_sorted) - 1):
        current_row = df_sorted.iloc[i][feature]
        current_y = df_sorted.iloc[i][target]
        next_row = df_sorted.iloc[i + 1][feature]
        next_y = df_sorted.iloc[i + 1][target]
        if current_row != next_row and current_y != next_y:
            midpoint = (current_row + next_row) / 2
            midpoints.add(midpoint)

    # now convert the set of midpoints to a list and sort them before returning
    midpoints_list = list(midpoints)
    midpoints_list.sort()

    return midpoints_list


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
class DecisionTreeClassifier:
    top_node = None
    theta = 0
    dtype_class = ['object', 'bool']

    def __init__(self, theta=0):
        self.theta = theta

    # Builds a classification tree
    def generate_tree(self, df, label_column, node):
        node_entropy = calc_entropy(df, label_column)

        # Find the highest gain feature and create a new node with this as data
        highest_gain_feature, highest_gain = find_highest_gain(df, label_column, node_entropy)
        node.data = highest_gain_feature
        node.df = df

        # If the node has 0 entropy or no feature with gain create a leaf with the most common class
        if node_entropy <= self.theta or highest_gain_feature is None:
            leaf_value = df[label_column].value_counts().idxmax()
            node.data = leaf_value
            node.df = None
            return

        # Determine if this feature is numerical or categorical
        if df[highest_gain_feature].dtype in self.dtype_class:
            node.type = "categorical"

            # Categorical - for each class in this feature, create a child node and call generate_tree for the child
            for highest_gain_feature_class in df[highest_gain_feature].unique():
                child_node = DecisionTreeClassNode(None, None)
                node.children[highest_gain_feature_class] = child_node
                df_branch = df.loc[df[highest_gain_feature] == highest_gain_feature_class].drop(
                    columns=[highest_gain_feature])
                self.generate_tree(df_branch, label_column, child_node)

            # Then create a "default" node containing the most common class for any feature classes that weren't trained
            most_common_class = df[label_column].value_counts().idxmax()
            child_node = DecisionTreeClassNode(most_common_class, None)
            node.children['default'] = child_node

        else:
            node.type = "numerical"

            # Numerical - find the midpoints between class changes for this feature, create a child nod for each
            midpoints = find_midpoints(df, highest_gain_feature, label_column)

            last_midpoint = df[highest_gain_feature].min()
            for midpoint in midpoints:
                child_node = DecisionTreeClassNode(None, None)
                node.children[midpoint] = child_node
                df_branch = df.loc[df[highest_gain_feature] >= last_midpoint].loc[df[highest_gain_feature] < midpoint]
                last_midpoint = midpoint
                self.generate_tree(df_branch.drop(columns=[highest_gain_feature]), label_column, child_node)

            # Now create a final child (called 'upper') for all entries >= the final midpoint
            child_node = DecisionTreeClassNode(None, None)
            node.children['upper'] = child_node
            df_branch = df.loc[df[highest_gain_feature] >= last_midpoint]
            self.generate_tree(df_branch.drop(columns=[highest_gain_feature]), label_column, child_node)

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
                        if verbose:
                            print(f"Taking branch <{midpoint}")
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
    def fit_predict_classify(self, df_trn, df_tst, label_column):
        self.top_node = DecisionTreeClassNode(None, None)
        self.generate_tree(df_trn, label_column, self.top_node)

        y = []
        for i in range(len(df_tst)):
            df_row = df_tst.iloc[i]
            y += [self.traverse_tree(df_tst, self.top_node)]
        return y

    # Makes predictions for an existing tree
    def predict_classify(self, df_tst):
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
    def prune_subtrees(self, subtrees, label_column, df_prune, verbose=False):
        prune_count = 0
        y_truth = df_prune[label_column].values.ravel()

        for subtree in subtrees:
            # First find the accuracy of the old tree
            y_pred = self.predict_classify(df_prune)
            score_pred = eval.eval_classification_score(y_truth, y_pred)
            if verbose:
                print(f"Accuracy of original tree= {score_pred:.2f}")

            # Replace the node with a leaf containing the most common class
            old_node = copy.deepcopy(subtree)
            subtree.children = {}
            subtree.type = None
            subtree.df = None
            subtree.data = old_node.df[label_column].value_counts().idxmax()

            # Now find the accuracy on the smaller tree
            y_pruned = self.predict_classify(df_prune)
            score_pruned = eval.eval_classification_score(y_truth, y_pruned)
            if verbose:
                print(f"Accuracy of tree with {old_node.data} pruned= {score_pruned:.2f}")

            # only if the pruned score < the original score, bring back the old node
            if score_pruned < score_pred:
                subtree.children = old_node.children
                subtree.type = old_node.type
                subtree.df = old_node.df
                subtree.data = old_node.data
            else:
                if verbose:
                    print(f"Pruned {old_node.data}")
                prune_count += 1


# Decision Tree Node
# Contains the data (a feature)
# children - a dictionary with feature values to nodes
# df - the dataset for this node
# type - whether the node's feature is a categorical or numerical value
class DecisionTreeClassNode:
    def __init__(self, data, df):
        self.data = data
        self.df = df
        self.type = None
        self.children = {}

    def find_child(self, child_data):
        return self.children[child_data]

import decision_tree_classifier
from decision_tree_classifier import DecisionTreeClassNode
from decision_tree_regressor import DecisionTreeRegNode
import eval
import pandas as pd


# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def classify_cross_validation(df_trn_partitions, model, label_column, df_prune=None):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # If not pruning, test the model and record its score
        if df_prune is None:
            y_pred = model.fit_predict_classify(df_trn_fold, df_test, label_column)
            y_truth = df_test[label_column].values.ravel()

        # If pruning first generate the full tree
        else:
            top_node = DecisionTreeClassNode(None, None)
            model.top_node = top_node
            model.generate_tree(df_trn_fold, label_column, top_node)

            # Now find the subtrees and prune
            subtrees = []
            for key in top_node.children.keys():
                if key != 'default' and key != 'upper':
                    model.find_subtrees(top_node.children[key], subtrees)
            model.prune_subtrees(subtrees, label_column, df_prune)

            # And then predict
            decision_tree_classifier.print_tree(top_node, None, None)
            y_pred = model.predict_classify(df_test)
            y_truth = df_test[label_column].values.ravel()

        # Print details about this fold
        score = eval.eval_classification_score(y_truth, y_pred)
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), Acc= {score*100:.2f}%')
        scores += [score]

    return scores


# Train a model on k-1 partitions and test on the remaining k-fold for each k-fold, returning an array of scores
def regression_cross_validation(df_trn_partitions, model, label_column, df_prune=None):
    scores = []
    k_folds = len(df_trn_partitions)

    for i in range(k_folds):

        # Construct a dataset for the model (k-1 partitions)
        df_trn_fold = pd.DataFrame(columns=df_trn_partitions[0].columns)
        parts_in_fold = []
        for j in range(k_folds):
            if i != j:
                df_trn_fold = pd.concat([df_trn_fold, df_trn_partitions[j]])
                parts_in_fold += [j]

        # Use the final, ith dataset for test
        df_test = df_trn_partitions[i]

        # If not pruning, test the model and record its score
        if df_prune is None:
            y_pred = model.fit_predict_regression(df_trn_fold, df_test, label_column)
            y_truth = df_test[label_column].values.ravel()

        # If pruning first generate the full tree
        else:
            top_node = DecisionTreeRegNode(None, None)
            model.top_node = top_node
            model.generate_tree(df_trn_fold, label_column, top_node)

            # Now find the subtrees and prune
            subtrees = []
            for key in top_node.children.keys():
                if key != 'default' and key != 'upper':
                    model.find_subtrees(top_node.children[key], subtrees)
            model.prune_subtrees(subtrees, label_column, df_prune)

            # And then predict
            y_pred = model.predict_regression(df_test)
            y_truth = df_test[label_column].values.ravel()

        # Print details about this fold
        score = eval.eval_mse(y_truth, y_pred)
        print(f'Fold {i+1}: Training on partitions {parts_in_fold} ({len(df_trn_fold)} entries), Testing on partition {i} ({len(df_test)} entries), MSE= {score:.2f}')
        scores += [score]

    return scores

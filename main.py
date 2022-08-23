# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import dataprep
import preprocessing
import decision_tree_classifier
import decision_tree_regressor
from decision_tree_classifier import DecisionTreeClassifier
from decision_tree_classifier import DecisionTreeClassNode
from decision_tree_regressor import DecisionTreeRegressor
from decision_tree_regressor import DecisionTreeRegNode
import processing
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Video Part 1
    print("VIDEO PART 1")

    # Show sample classification tree w/o pruning
    # At the top, the tree splits on 'Uniformity of Cell Shape'
    # For the branch <3.5 there is a second split on 'Normal Nucleoli'
    # All other branches end in leaves
    df_breast_sample = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_breast_sample = df_breast_sample.iloc[0:30]
    dt = DecisionTreeClassifier()
    top_node = DecisionTreeClassNode(None, None)
    dt.generate_tree(df_breast_sample, 'Class', top_node)
    dt.top_node = top_node
    print("CLASSIFICATION TREE W/O PRUNING - BREAST CANCER SAMPLE")
    decision_tree_classifier.print_tree(top_node, None, None)
    print("==============================\n")

    # Show sample classification tree w/ pruning
    # The 'Normal Nucleoli' branch is pruned away
    # The only split will be on 'Uniformity of Cell Shape'
    df_breast_sample_test = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_breast_sample_test = df_breast_sample_test.iloc[31:60]
    subtrees = []
    for key in top_node.children.keys():
        if key != 'default' and key != 'upper':
            dt.find_subtrees(top_node.children[key], subtrees)
    dt.prune_subtrees(subtrees, 'Class', df_breast_sample_test)
    print("CLASSIFICATION TREE W/ PRUNING - BREAST CANCER SAMPLE")
    decision_tree_classifier.print_tree(top_node, None, None)
    print("==============================\n")

    # Show sample regression tree w/o pruning
    # The tree will split on vendor name, MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, and ERP
    df_machine_sample = dataprep.prepare_machine('datasets/machine.data')
    df_machine_sample = df_machine_sample.iloc[0:3]
    dt = DecisionTreeRegressor()
    top_node = DecisionTreeRegNode(None, None)
    dt.generate_tree(df_machine_sample, 'PRP', top_node)
    dt.top_node = top_node
    print("REGRESSION TREE W/O PRUNING - MACHINE SAMPLE")
    decision_tree_regressor.print_tree(top_node, None, None)
    print("==============================\n")

    # Show sample regression tree w/ pruning
    # After pruning, the tree will only split along vendor name
    df_machine_sample_test = dataprep.prepare_machine('datasets/machine.data')
    df_machine_sample_test = df_machine_sample_test.iloc[4:5]
    subtrees = []
    for key in top_node.children.keys():
        if key != 'default' and key != 'upper':
            dt.find_subtrees(top_node.children[key], subtrees)
    dt.prune_subtrees(subtrees, 'PRP', df_machine_sample_test)
    print("CLASSIFICATION TREE W/ PRUNING - MACHINE SAMPLE")
    decision_tree_classifier.print_tree(top_node, None, None)
    print("==============================\n")

    # Video Part 2
    print("VIDEO PART 2")

    # Demonstrate calculation of information gain
    entropy = decision_tree_classifier.calc_entropy(df_breast_sample, 'Class')
    print(f"Entropy for breast cancer sample= {entropy:.2f}")

    # Demonstrate gain ratio calculation
    feature_entropy = decision_tree_classifier.calc_feature_entropy(df_breast_sample, 'Uniformity of Cell Shape',
                                                                    'Class')
    print(f"Feature entropy for Uniformity of Cell Shape= {feature_entropy:.2f}")

    # Demonstrate MSE calculation
    mse = decision_tree_regressor.calc_error(df_machine_sample, 'PRP')
    print(f"MSE for machine sample= {mse:.2f}")
    print("==============================\n")

    # Video Part 3
    print("VIDEO PART 3")

    # Demonstrate decision made to prune subtree
    dt = DecisionTreeClassifier()
    top_node = DecisionTreeClassNode(None, None)
    dt.generate_tree(df_breast_sample, 'Class', top_node)
    dt.top_node = top_node

    subtrees = []
    for key in top_node.children.keys():
        if key != 'default' and key != 'upper':
            dt.find_subtrees(top_node.children[key], subtrees)
    dt.prune_subtrees(subtrees, 'Class', df_breast_sample_test, verbose=True)
    print("==============================\n")

    # Video Part 4
    print("VIDEO PART 4")

    # Demonstrate traversal of a classification tree for the unpruned breast cancer sample
    dt = DecisionTreeClassifier()
    top_node = DecisionTreeClassNode(None, None)
    dt.generate_tree(df_breast_sample, 'Class', top_node)
    dt.top_node = top_node

    df_breast_sample_test = df_breast_sample_test.iloc[0:1]
    print(df_breast_sample_test)
    print()
    dt.traverse_tree(df_breast_sample_test, top_node, verbose=True)
    print("==============================\n")

    # Video Part 5
    print("VIDEO PART 5")

    # Demonstrate traversal of a regression tree for the unpruned machine sample
    dt = DecisionTreeRegressor()
    top_node = DecisionTreeRegNode(None, None)
    dt.generate_tree(df_machine_sample, 'PRP', top_node)
    dt.top_node = top_node

    df_machine_sample_test = df_machine_sample_test.iloc[0:1]
    print(df_machine_sample_test)
    print()
    dt.traverse_tree(df_machine_sample_test, top_node, verbose=True)
    print("==============================\n")

    # Video Part 6
    print("VIDEO PART 6")

    # perform 5-fold cv on abalone
    print("ABALONE - UNPRUNED")
    df_abalone = dataprep.prepare_abalone('datasets/abalone.data')
    df_abalone_partitions = preprocessing.df_partition(df_abalone, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_abalone_partitions, dt, 'Rings')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on abalone, but also prune
    print("ABALONE - PRUNED")
    df_abalone = dataprep.prepare_abalone('datasets/abalone.data')
    df_abalone_trn, df_abalone_prune = preprocessing.df_split(df_abalone, 'Rings', 0.8)
    df_abalone_partitions = preprocessing.df_partition(df_abalone_trn, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_abalone_partitions, dt, 'Rings', df_abalone_prune)
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    print("==============================\n")

    # perform 5-fold cv on car
    print("CAR - UNPRUNED")
    df_car = dataprep.prepare_car('datasets/car.data')
    df_car_partitions = preprocessing.df_stratified_partition(df_car, 'CAR', 5)
    dt = DecisionTreeClassifier(theta=0)
    scores = processing.classify_cross_validation(df_car_partitions, dt, 'CAR')
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")

    # perform 5-fold cv on car, but also prune
    print("CAR - PRUNED")
    df_car = dataprep.prepare_car('datasets/car.data')
    df_car_trn, df_car_prune = preprocessing.df_stratified_split(df_car, 'CAR', 0.8)
    df_car_partitions = preprocessing.df_stratified_partition(df_car_trn, 'CAR', 5)
    dt = DecisionTreeClassifier(theta=0)
    scores = processing.classify_cross_validation(df_car_partitions, dt, 'CAR', df_car_prune)
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")

    print("==============================\n")

    # perform 5-fold cv on forest fires
    print("FOREST FIRES - UNPRUNED")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_partitions = preprocessing.df_partition(df_forest, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_forest_partitions, dt, 'area')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on forest fires, but also prune
    print("FOREST FIRES - PRUNED")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_trn, df_forest_prune = preprocessing.df_split(df_forest, 'area', 0.8)
    df_forest_partitions = preprocessing.df_partition(df_forest_trn, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_forest_partitions, dt, 'area', df_forest_prune)
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    print("==============================\n")

    # perform 5-fold cv on house
    print("HOUSE VOTES - UNPRUNED")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house_partitions = preprocessing.df_stratified_partition(df_house, 'party', 5)
    dt = DecisionTreeClassifier(theta=0)
    scores = processing.classify_cross_validation(df_house_partitions, dt, 'party')
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")

    # perform 5-fold cv on house, but also prune
    print("HOUSE VOTES - PRUNED")
    df_house = dataprep.prepare_house_votes_84('datasets/house-votes-84.data')
    df_house_trn, df_house_prune = preprocessing.df_stratified_split(df_house, 'party', 0.8)
    df_house_partitions = preprocessing.df_stratified_partition(df_house_trn, 'party', 5)
    dt = DecisionTreeClassifier(theta=0)
    scores = processing.classify_cross_validation(df_house_partitions, dt, 'party', df_house_prune)
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")

    print("==============================\n")

    # perform 5-fold cv on machine
    print("MACHINE - UNPRUNED")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_partitions = preprocessing.df_partition(df_machine, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_machine_partitions, dt, 'PRP')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on machine, but also prune
    print("MACHINE - PRUNED")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_trn, df_machine_prune = preprocessing.df_split(df_machine, 'PRP', 0.8)
    df_machine_partitions = preprocessing.df_partition(df_machine_trn, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_machine_partitions, dt, 'PRP', df_machine_prune)
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    print("==============================\n")

    # perform 5-fold cv on breast
    print("BREAST CANCER - UNPRUNED")
    df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_breast_partitions = preprocessing.df_stratified_partition(df_breast, 'Class', 5)
    dt = DecisionTreeClassifier(theta=0)
    scores = processing.classify_cross_validation(df_breast_partitions, dt, 'Class')
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")

    # perform 5-fold cv on breast, but also prune
    print("BREAST CANCER - PRUNED")
    df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
    df_breast_trn, df_breast_prune = preprocessing.df_stratified_split(df_breast, 'Class', 0.8)
    df_breast_partitions = preprocessing.df_stratified_partition(df_breast_trn, 'Class', 5)
    dt = DecisionTreeClassifier(theta=0)
    scores = processing.classify_cross_validation(df_breast_partitions, dt, 'Class', df_breast_prune)
    print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")

    # Try to improve forest fires using median calculation
    # perform 5-fold cv on forest fires
    print("FOREST FIRES (MEAN) - UNPRUNED")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_partitions = preprocessing.df_partition(df_forest, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_forest_partitions, dt, 'area')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on forest fires, but also prune
    print("FOREST FIRES (MEAN) - PRUNED")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_trn, df_forest_prune = preprocessing.df_split(df_forest, 'area', 0.8)
    df_forest_partitions = preprocessing.df_partition(df_forest_trn, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_forest_partitions, dt, 'area', df_forest_prune)
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on forest fires
    print("FOREST FIRES (MEDIAN) - UNPRUNED")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_partitions = preprocessing.df_partition(df_forest, 5)
    dt = DecisionTreeRegressor(theta=0, mean_or_median='median')
    scores = processing.regression_cross_validation(df_forest_partitions, dt, 'area')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on forest fires, but also prune
    print("FOREST FIRES (MEDIAN) - PRUNED")
    df_forest = dataprep.prepare_forestfires('datasets/forestfires.data')
    df_forest_trn, df_forest_prune = preprocessing.df_split(df_forest, 'area', 0.8)
    df_forest_partitions = preprocessing.df_partition(df_forest_trn, 5)
    dt = DecisionTreeRegressor(theta=0, mean_or_median='median')
    scores = processing.regression_cross_validation(df_forest_partitions, dt, 'area', df_forest_prune)
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # Try to improve machine using median calculation
    # perform 5-fold cv on machine
    print("MACHINE (MEAN) - UNPRUNED")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_partitions = preprocessing.df_partition(df_machine, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_machine_partitions, dt, 'PRP')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on machine, but also prune
    print("MACHINE (MEAN) - PRUNED")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_trn, df_machine_prune = preprocessing.df_split(df_machine, 'PRP', 0.8)
    df_machine_partitions = preprocessing.df_partition(df_machine_trn, 5)
    dt = DecisionTreeRegressor(theta=0)
    scores = processing.regression_cross_validation(df_machine_partitions, dt, 'PRP', df_machine_prune)
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on machine
    print("MACHINE (MEDIAN) - UNPRUNED")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_partitions = preprocessing.df_partition(df_machine, 5)
    dt = DecisionTreeRegressor(theta=0, mean_or_median='median')
    scores = processing.regression_cross_validation(df_machine_partitions, dt, 'PRP')
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")

    # perform 5-fold cv on machine, but also prune
    print("MACHINE (MEDIAN) - PRUNED")
    df_machine = dataprep.prepare_machine('datasets/machine.data')
    df_machine_trn, df_machine_prune = preprocessing.df_split(df_machine, 'PRP', 0.8)
    df_machine_partitions = preprocessing.df_partition(df_machine_trn, 5)
    dt = DecisionTreeRegressor(theta=0, mean_or_median='median')
    scores = processing.regression_cross_validation(df_machine_partitions, dt, 'PRP', df_machine_prune)
    print(f"Avg 5-Fold CV MSE: {np.mean(scores):.2f}")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

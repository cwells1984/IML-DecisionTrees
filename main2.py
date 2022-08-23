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

print("BREAST CANCER - PRUNED")
df_breast = dataprep.prepare_breast_cancer_wisconsin('datasets/breast-cancer-wisconsin.data')
df_breast_trn, df_breast_prune = preprocessing.df_stratified_split(df_breast, 'Class', 0.8)
df_breast_partitions = preprocessing.df_stratified_partition(df_breast_trn, 'Class', 5)
dt = DecisionTreeClassifier(theta=0)
scores = processing.classify_cross_validation(df_breast_partitions, dt, 'Class', df_breast_prune)
print(f"Avg 5-Fold CV Accuracy: {np.mean(scores) * 100:.2f}%")
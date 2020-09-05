import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor\
                            , VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from utils import rgb2gray, get_regression_data, visualise_regression_data
from assigning_library import read_csv_files, split_train_validation, split_features_labels

path_train_file = "../Data/winequality-red-train.csv"
path_test_file = "../Data/winequality-red-test.csv"

my_train_File, my_test_File = read_csv_files(path_train_file, path_test_file)
                                                                       
header, X_train, X_validation, Y_train, Y_validation = split_train_validation(my_train_File, split_rate=0.25)

X_test, Y_test = split_features_labels(my_test_File)
                                                                                    
###################################################################################################################
############################### This part is using Random Forest in Regression form ###############################
###################################################################################################################
best_regr_score_randomforest = 0
progress_randomforest = 0
regr_score_randomforest = 0

n_estimator_strarting_point = 100
n_estimator_ending_point = 500
n_estimator_step_size = 100   

max_features_starting_point = 2
max_features_ending_point = 10
max_features_step_size = 2     

max_depth_starting_point = 3
max_depth_ending_point = 40
max_depth_step_size = 4           

number_of_n_estimators \
= int((n_estimator_ending_point + n_estimator_step_size - n_estimator_strarting_point-1)/n_estimator_step_size)
number_of_max_features \
= int((max_features_ending_point + max_features_step_size - max_features_starting_point-1)/max_features_step_size)
number_of_max_depth \
= int((max_depth_ending_point + max_depth_step_size - max_depth_starting_point-1)/max_depth_step_size)

total_number_of_randomforest_models = number_of_n_estimators * number_of_max_features * number_of_max_depth

for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for max_features_idx in range(max_features_starting_point, max_features_ending_point, max_features_step_size):      
        for max_depth_idx in range(max_depth_starting_point, max_depth_ending_point, max_depth_step_size):  
            os.system('cls')
            print("Random Forest Progress : ",int(progress_randomforest),"%")
            print("Number of estimator of {} Maximum features of {} and Maximum depth of {} \
                    gives accuracy score of {:.2f}%".format(n_estimator_idx, max_features_idx, \
                    max_depth_idx, regr_score_randomforest*100))
            progress_randomforest += 100/total_number_of_randomforest_models
            randomforestregressor = RandomForestRegressor(n_estimators = n_estimator_idx,  \
                                    max_features = max_features_idx, max_depth = max_depth_idx)
            randomforestregressor.fit(X_train, Y_train)  
            Y_prediction = np.around(randomforestregressor.predict(X_validation))  
            regr_score_randomforest = accuracy_score(Y_validation, Y_prediction)
            if regr_score_randomforest > best_regr_score_randomforest:  
                best_n_estimators_randomforest = n_estimator_idx  
                best_max_features_randomforest = max_features_idx 
                best_max_depth_randomforest = max_depth_idx  
                best_regr_score_randomforest = regr_score_randomforest 
os.system('cls')
print("\n              Random Forest performance ")
print("            --------------------------------")
print("Best Regressor Score for Random Forest : {:.2f}%".format(best_regr_score_randomforest*100)) 
print("Best Estimator number : ", best_n_estimators_randomforest, "\nBest Features number : "\
      , best_max_features_randomforest, "\nbest_max_depth : ",best_max_depth_randomforest)

input("Press Enter to continue...")

###################################################################################################################




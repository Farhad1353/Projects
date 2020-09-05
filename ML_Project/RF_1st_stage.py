import sys                    # import libraries
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor\
                            , VotingRegressor, StackingRegressor # import sklearn libraries
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score

from utils import rgb2gray, get_regression_data, visualise_regression_data # import our libraries
from assigning_library import read_csv_files, split_train_validation, split_features_labels

#######################################################################################################################
## This is Part 1 of the project where we read the train and test csv files and splittig them to features and labels ##
#######################################################################################################################

path_train_file = "../Data/winequality-red-train.csv" # assigning the train.csv file path to a variable
path_test_file = "../Data/winequality-red-test.csv" # assigning the test.csv file path to a variable

my_train_File, my_test_File = read_csv_files(path_train_file, path_test_file) # reading the csv files using pandas
                                                                       
header, X_train, X_validation, Y_train, Y_validation\
 = split_train_validation(my_train_File, split_rate=0.25) # splitting the train.csv to train and validation arrays

X_test, Y_test = split_features_labels(my_test_File) # splitting the test.csv file to features and label arrays
                                                                                    
###################################################################################################################
############## This part is using Random Forest in Regression form(Part 2 of the Project) #########################
###################################################################################################################
best_regr_score_randomforest = 0 # variable which shows the best scoring RandomForest model
progress_randomforest = 0  # variable which indicates how far have we processed our RandomForest models
regr_score_randomforest = 0 # variable which shows the score for each of the Random Forest model

n_estimator_strarting_point = 100   # setting the range for n_estimators for different RandomForest models
n_estimator_ending_point = 500
n_estimator_step_size = 100   

max_features_starting_point = 2   # setting the range for max_features for different RandomForest models
max_features_ending_point = 10
max_features_step_size = 2     

max_depth_starting_point = 3    # setting the range for max_depth for different RandomForest models
max_depth_ending_point = 40
max_depth_step_size = 4           

number_of_n_estimators \
= int((n_estimator_ending_point + n_estimator_step_size - n_estimator_strarting_point-1)/n_estimator_step_size)
number_of_max_features \
= int((max_features_ending_point + max_features_step_size - max_features_starting_point-1)/max_features_step_size)
number_of_max_depth \
= int((max_depth_ending_point + max_depth_step_size - max_depth_starting_point-1)/max_depth_step_size)
                   # calculating the total number of RandomForest models
total_number_of_randomforest_models = number_of_n_estimators * number_of_max_features * number_of_max_depth

                ### training and testing(through validation sets) all the RandomForest models
for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for max_features_idx in range(max_features_starting_point, max_features_ending_point, max_features_step_size):      
        for max_depth_idx in range(max_depth_starting_point, max_depth_ending_point, max_depth_step_size):  
            os.system('cls')
            print("Random Forest Progress : ",int(progress_randomforest),"%") # showing the progress of the process
            print("Number of estimators of {}      Maximum features of {} and Maximum depth of {} \
                    gives accuracy score of {:.2f}%".format(n_estimator_idx, max_features_idx, \
                    max_depth_idx, regr_score_randomforest*100))  # score of one single RandomForest model
            progress_randomforest += 100/total_number_of_randomforest_models
            randomforestregressor = RandomForestRegressor(n_estimators = n_estimator_idx,  \
            max_features = max_features_idx, max_depth = max_depth_idx) # assigning an instance of a RandomForest
            randomforestregressor.fit(X_train, Y_train)  # fitting the train arrays to each RandomForest model
            Y_prediction = np.around(randomforestregressor.predict(X_validation)) #predicting using the validation set  
            regr_score_randomforest = accuracy_score(Y_validation, Y_prediction) #accuaracy score for each model
            if regr_score_randomforest > best_regr_score_randomforest:  # assessing if the new score is better 
                best_n_estimators_randomforest = n_estimator_idx        # than the previous best score
                best_max_features_randomforest = max_features_idx 
                best_max_depth_randomforest = max_depth_idx  
                best_regr_score_randomforest = regr_score_randomforest 
os.system('cls')
print("\n              Random Forest performance ")
print("            --------------------------------")  # show the best model of RandomForest tested on validation set
print("Best Regressor Score for Random Forest : {:.2f}%".format(best_regr_score_randomforest*100)) 
print("Best Estimator number : ", best_n_estimators_randomforest, "\nBest Features number : "\
      , best_max_features_randomforest, "\nbest_max_depth : ",best_max_depth_randomforest)

randomforest_best_parameters = np.array([best_n_estimators_randomforest\
                              , best_max_features_randomforest, best_max_depth_randomforest])

np.savetxt('../Data/RandomForest_reg.csv', randomforest_best_parameters, fmt="%d", delimiter=",")

input("Press Enter to continue...")

###################################################################################################################




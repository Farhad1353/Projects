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
################### This part is using AdaBoost in Regression form(Part 2 of the Project) #########################
###################################################################################################################
best_regr_score_adaboost = 0 # variable which shows the best scoring AdaBoost model
progress_adaboost = 0  # variable which indicates how far have we processed our AdaBoost models
regr_score_adaboost = 0 # variable which shows the score for each of the AdaBoost model

n_estimator_strarting_point = 10   # setting the range for n_estimators for different AdaBoost models
n_estimator_ending_point = 100
n_estimator_step_size = 10   

learning_rate_starting_point = 5   # setting the range for learning_rates for different AdaBoost models
learning_rate_ending_point = 100
learning_rate_step_size = 5               

number_of_n_estimators \
= int((n_estimator_ending_point + n_estimator_step_size - n_estimator_strarting_point-1)/n_estimator_step_size)
number_of_learning_rates \
= int((learning_rate_ending_point + learning_rate_step_size - learning_rate_starting_point-1)/learning_rate_step_size)
                   # calculating the total number of AdaBoost models
total_number_of_adaboost_models = number_of_n_estimators * number_of_learning_rates

                ### training and testing(through validation sets) all the AdaBoost models
for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for learning_rate_idx in range(learning_rate_starting_point, learning_rate_ending_point, learning_rate_step_size):    
        os.system('cls')
        print("Ada Boost Progress : ",int(progress_adaboost),"%") # showing the progress of the process
        print("Number of estimators of {} and Learning rate of {}\
        gives accuracy score of {:.2f}%".format(n_estimator_idx, learning_rate_idx/100,\
        regr_score_adaboost*100))  # score of one single AdaBoost model
        progress_adaboost += 100/total_number_of_adaboost_models
        adaboostregressor = AdaBoostRegressor(n_estimators = n_estimator_idx,  \
        learning_rate = learning_rate_idx/100) # assigning an instance of a AdaBoost
        adaboostregressor.fit(X_train, Y_train)  # fitting the train arrays to each AdaBoost model
        Y_prediction = np.around(adaboostregressor.predict(X_validation)) #predicting using the validation set  
        regr_score_adaboost = accuracy_score(Y_validation, Y_prediction) #accuaracy score for each model
        if regr_score_adaboost > best_regr_score_adaboost:  # assessing if the new score is better 
            best_n_estimators_adaboost = n_estimator_idx        # than the previous best score
            best_learning_rate_adaboost = learning_rate_idx   
            best_regr_score_adaboost = regr_score_adaboost 
os.system('cls')
print("\n                Ada Boost performance ")
print("            --------------------------------")  # show the best model of AdaBoost tested on validation set
print("Best Regressor Score for Ada Boost : {:.2f}%".format(best_regr_score_adaboost*100)) 
print("Best Estimator number : ", best_n_estimators_adaboost, "\nBest Learning : "\
      , best_learning_rate_adaboost/100)

adaboost_best_parameters = np.array([best_n_estimators_adaboost, best_learning_rate_adaboost])

np.savetxt('../Data/AdaBoost_reg.csv', adaboost_best_parameters, fmt="%d", delimiter=",")

input("Press Enter to continue...")

###################################################################################################################




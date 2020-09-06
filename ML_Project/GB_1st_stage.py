import sys                    # import libraries
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import GradientBoostingRegressor # import sklearn libraries 
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
############## This part is using GradientBoosting in Regression form(Part 2 of the Project) #########################
###################################################################################################################
best_regr_score_gradientboosting = 0 # variable which shows the best scoring GradientBoosting model
progress_gradientboosting = 0  # variable which indicates how far have we processed our GradientBoosting models
regr_score_gradientboosting = 0 # variable which shows the score for each of the GradientBoosting model

n_estimator_strarting_point = 60   # setting the range for n_estimators for different GradientBoosting models
n_estimator_ending_point = 130
n_estimator_step_size = 10   

learning_rate_starting_point = 3   # setting the range for learning_rate for different GradientBoosting models
learning_rate_ending_point = 50
learning_rate_step_size = 5     

max_depth_starting_point = 3    # setting the range for max_depth for different GradientBoosting models
max_depth_ending_point = 10
max_depth_step_size = 2           

number_of_n_estimators \
= int((n_estimator_ending_point + n_estimator_step_size - n_estimator_strarting_point-1)/n_estimator_step_size)
number_of_learning_rate \
= int((learning_rate_ending_point + learning_rate_step_size - learning_rate_starting_point-1)/learning_rate_step_size)
number_of_max_depth \
= int((max_depth_ending_point + max_depth_step_size - max_depth_starting_point-1)/max_depth_step_size)
                   # calculating the total number of GradientBoosting models
total_number_of_gradientboosting_models = number_of_n_estimators * number_of_learning_rate * number_of_max_depth

                ### training and testing(through validation sets) all the GradientBoosting models
for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for learning_rate_idx in range(learning_rate_starting_point, learning_rate_ending_point, learning_rate_step_size):      
        for max_depth_idx in range(max_depth_starting_point, max_depth_ending_point, max_depth_step_size):  
            os.system('cls')
            print("Gradient Boosting Progress : ",int(progress_gradientboosting),"%") # showing the progress of the process
            print("Number of estimators of {}      Learning rate of {} and Maximum depth of {} \
                    gives accuracy score of {:.2f}%".format(n_estimator_idx, learning_rate_idx/100, \
                    max_depth_idx, regr_score_gradientboosting*100))  # score of one single GradientBoosting model
            progress_gradientboosting += 100/total_number_of_gradientboosting_models
            gradientboostingregressor = GradientBoostingRegressor(n_estimators = n_estimator_idx, learning_rate\
                 = learning_rate_idx/100, max_depth = max_depth_idx) # assigning an instance of a GradientBoosting
            gradientboostingregressor.fit(X_train, Y_train)  # fitting the train arrays to each GradientBoosting model
            Y_prediction = np.around(gradientboostingregressor.predict(X_validation)) #predicting using the validation set  
            regr_score_gradientboosting = accuracy_score(Y_validation, Y_prediction) #accuaracy score for each model
            if regr_score_gradientboosting > best_regr_score_gradientboosting:  # assessing if the new score is better 
                best_n_estimators_gradientboosting = n_estimator_idx        # than the previous best score
                best_learning_rate_gradientboosting = learning_rate_idx 
                best_max_depth_gradientboosting = max_depth_idx  
                best_regr_score_gradientboosting = regr_score_gradientboosting 
os.system('cls')
print("\n              Gradient Boosting performance ")
print("            --------------------------------")  # show the best model of GradientBoosting tested on validation set
print("Best Regressor Score for Gradient Boosting : {:.2f}%".format(best_regr_score_gradientboosting*100)) 
print("Best Estimator number : ", best_n_estimators_gradientboosting, "\nBest Learning rate : "\
      , best_learning_rate_gradientboosting/100, "\nbest_max_depth : ",best_max_depth_gradientboosting)

gradientboosting_best_parameters = np.array([best_n_estimators_gradientboosting\
                              , best_learning_rate_gradientboosting, best_max_depth_gradientboosting])

np.savetxt('../Data/GradientBoosting_reg.csv', gradientboosting_best_parameters, fmt="%d", delimiter=",")

input("Press Enter to continue...")

###################################################################################################################




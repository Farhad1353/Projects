import sys                     # import libraries
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
                         
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor\
                            , VotingRegressor, StackingRegressor     # import sklearn libraries 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score

from utils import rgb2gray, get_regression_data, visualise_regression_data # import our libraries
from assigning_library import read_csv_files, split_train_validation, split_features_labels, read_param
from regression_library import low_high_param, show_features_impact, plot_models

#######################################################################################################################
## This is Part 1 of the project where we read the train and test csv files and splittig them to features and labels ##
#######################################################################################################################

path_train_file = "../Data/winequality-red-train.csv"  # assigning the train.csv file path to a variable
path_test_file = "../Data/winequality-red-test.csv"  # assigning the test.csv file path to a variable

my_train_File, my_test_File = read_csv_files(path_train_file, path_test_file)  # reading the csv files using pandas
                                                                       
header, X_train, X_validation, Y_train, Y_validation\
= split_train_validation(my_train_File, split_rate=0.25)  # splitting the train.csv to train and validation arrays

X_test, Y_test = split_features_labels(my_test_File) # splitting the test.csv file to features and label arrays
                                                                                    
###################################################################################################################
############## This part is using Random Forest in Regression form(Parts 2 & 3 of the Project) ####################
###################################################################################################################

best_regr_score_randomforest = 0  # variable which shows the best scoring RandomForest model
progress_randomforest = 0   # variable which indicates how far have we processed our RandomForest models
regr_score_randomforest = 0 # variable which shows the score for each of the Random Forest model

n_estimator_midpoint, max_features_midpoint, max_depth_midpoint\
                = read_param("../Data/RandomForest_reg.csv", 3) # reading the parameters from 1st stage of RandomForest
n_estimator_step_size = 50 # defining the step change of n_estimator moving from one model to another
n_estimator_strarting_point, n_estimator_ending_point = low_high_param\
(n_estimator_midpoint, n_estimator_step_size, 3) # finding lowest and highest values for n_estimators of RandomForest

max_features_step_size = 1 # defining the step change of max_features moving from one model to another
max_features_starting_point, max_features_ending_point = low_high_param\
(max_features_midpoint,max_features_step_size, 3) # finding lowest and highest values for max_features of RandomForest
    
max_depth_step_size = 2 # defining the step change of max_depth moving from one model to another
max_depth_starting_point, max_depth_ending_point = low_high_param\
(max_depth_midpoint,max_depth_step_size, 3) # finding lowest and highest values for depth_features of RandomForest

randomforest_models = np.zeros((27,2)) # defining an array for accuracy scores
model_idx = 0     # index for each model                           
best_model_score = 0    # variable for the best score

                ### training and testing(through validation sets) all the RandomForest models
for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for max_features_idx in range(max_features_starting_point, max_features_ending_point, max_features_step_size):      
        for max_depth_idx in range(max_depth_starting_point, max_depth_ending_point, max_depth_step_size):  
            os.system('cls')
            print("Random Forest Progress : ",int(progress_randomforest),"%")  # showing the progress of the process
            print("Number of estimator of {} Maximum features of {} and Maximum depth of {} \
                    gives accuracy score of {:.2f}%".format(n_estimator_idx, max_features_idx, \
                    max_depth_idx, regr_score_randomforest*100))  # score of one single RandomForest model
            print("Best Score so far : ", round(best_regr_score_randomforest*100,2), "%")
            progress_randomforest += 100/27
            randomforestregressor = RandomForestRegressor(n_estimators = n_estimator_idx,  \
                max_features = max_features_idx, max_depth = max_depth_idx) # assigning an instance of a RandomForest
            randomforestregressor.fit(X_train, Y_train)  # fitting the train arrays to each RandomForest model
            Y_prediction = np.around(randomforestregressor.predict(X_validation)) #predicting using the validation set
            regr_score_randomforest = accuracy_score(Y_validation, Y_prediction) #accuaracy score for each model
            randomforest_models[model_idx, 1] = regr_score_randomforest * 100 # saving the score for each model
            randomforest_models[model_idx, 0] = model_idx                 # saving the index for each model
            if regr_score_randomforest > best_regr_score_randomforest: # assessing if the new score is better 
                best_n_estimators_randomforest = n_estimator_idx      # than the previous best score
                best_max_features_randomforest = max_features_idx 
                best_max_depth_randomforest = max_depth_idx  
                best_regr_score_randomforest = regr_score_randomforest
                best_model_score = randomforest_models[model_idx, 1]
                best_model_idx = randomforest_models[model_idx, 0]
            model_idx+=1 
os.system('cls')
print("\n               RandomForest performance ")
print("            --------------------------------") # show the best model of RandomForest tested on validation set
print("Best Regressor Score for Random Forest : {:.2f}%".format(best_regr_score_randomforest*100)) 
print("Best Estimator number : ", best_n_estimators_randomforest, "\nBest Features number : "\
      , best_max_features_randomforest, "\nbest_max_depth : ",best_max_depth_randomforest)

plot_models(randomforest_models[:,0], randomforest_models[:,1], "RandomForest-Models", 'g') # Plot all the models

print("Best model : model ",int(best_model_idx), " with score of ", round(best_model_score,2), "%" )

input("Press Enter to continue...")

###################################################################################################################
################### This part is using AdaBoost in Regression form(Parts 2 & 3 of the Project) ####################
###################################################################################################################

best_regr_score_adaboost = 0 # variable which shows the best scoring AdaBoost model
progress_adaboost = 0  # variable which indicates how far have we processed our AdaBoost models
regr_score_adaboost = 0 # variable which shows the score for each of the AdaBoost model

n_estimator_midpoint, learning_rate_midpoint = read_param("../Data/AdaBoost_reg.csv", 2) # reading the parameters from 1st stage of AdaBoost
n_estimator_step_size = 4 # defining the step change of n_estimator moving from one model to another
n_estimator_strarting_point, n_estimator_ending_point = low_high_param\
(n_estimator_midpoint, n_estimator_step_size, 2) # finding lowest and highest values for n_estimators of AdaBoost

learning_rate_step_size = 3 # defining the step change of max_features moving from one model to another
learning_rate_starting_point, learning_rate_ending_point = low_high_param\
(learning_rate_midpoint, learning_rate_step_size, 2) # finding lowest and highest values for learning_rate of AdaBoost
adaboost_models = np.zeros((25,2)) # defining an array for accuracy scores
model_idx = 0     # index for each model                           
best_model_score = 0    # variable for the best score

                ### training and testing(through validation sets) all the AdaBoost models
for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for learning_rate_idx in range(learning_rate_starting_point, learning_rate_ending_point, learning_rate_step_size):       
        os.system('cls')
        print("AdaBoost Progress : ",int(progress_adaboost),"%")  # showing the progress of the process
        print("Number of estimator of {}      Learning Rate of {} \
        gives accuracy score of {:.2f}%".format(n_estimator_idx, learning_rate_idx/100,\
        regr_score_adaboost*100))  # score of one single AdaBoost model
        print("Best Score so far : ", round(best_regr_score_adaboost*100,2), "%")
        progress_adaboost += 100/25
        adaboostregressor = AdaBoostRegressor(n_estimators = n_estimator_idx,\
        learning_rate = learning_rate_idx/100) # assigning an instance of a AdaBoost
        adaboostregressor.fit(X_train, Y_train)  # fitting the train arrays to each AdaBoost model
        Y_prediction = np.around(adaboostregressor.predict(X_validation)) #predicting using the validation set
        regr_score_adaboost = accuracy_score(Y_validation, Y_prediction) #accuaracy score for each model
        adaboost_models[model_idx, 1] = regr_score_adaboost * 100 # saving the score for each model
        adaboost_models[model_idx, 0] = model_idx    # saving the index for each model
        if regr_score_adaboost > best_regr_score_adaboost: # assessing if the new score is better 
            best_n_estimators_adaboost = n_estimator_idx      # than the previous best score
            best_learning_rate_adaboost = learning_rate_idx  
            best_regr_score_adaboost = regr_score_adaboost
            best_model_score = adaboost_models[model_idx, 1]
            best_model_idx = adaboost_models[model_idx, 0]
        model_idx+=1
os.system('cls')
print("\n                  AdaBoost performance ")
print("            --------------------------------") # show the best model of AdaBoost tested on validation set
print("Best Regressor Score for AdaBoost : {:.2f}%".format(best_regr_score_adaboost*100)) 
print("Best Estimator number : ", best_n_estimators_adaboost, "\nBest Learning rate : "\
      , best_learning_rate_adaboost/100)

plot_models(adaboost_models[:,0], adaboost_models[:,1], "AdaBoost-Models", 'b') # Plot all the models

print("Best model : model ",int(best_model_idx), " with score of ", round(best_model_score,2), "%" )

input("Press Enter to continue...")

###################################################################################################################
############## This part is using Gradientboosting in Regression form(Parts 2 & 3 of the Project) #################
###################################################################################################################

best_regr_score_gradientboosting = 0  # variable which shows the best scoring Gradientboosting model
progress_gradientboosting = 0   # variable which indicates how far have we processed our Gradientboosting models
regr_score_gradientboosting = 0 # variable which shows the score for each of the Gradientboosting model

n_estimator_midpoint, learning_rate_midpoint, max_depth_midpoint\
                = read_param("../Data/Gradientboosting_reg.csv", 3) # reading the parameters from 1st stage of Gradientboosting
n_estimator_step_size = 5 # defining the step change of n_estimator moving from one model to another
n_estimator_strarting_point, n_estimator_ending_point = low_high_param\
(n_estimator_midpoint, n_estimator_step_size, 3) # finding lowest and highest values for n_estimators of Gradientboosting

learning_rate_step_size = 2 # defining the step change of learning_rate moving from one model to another
learning_rate_starting_point, learning_rate_ending_point = low_high_param\
(learning_rate_midpoint,learning_rate_step_size, 3) # finding lowest and highest values for learning_rate of Gradientboosting
    
max_depth_step_size = 1 # defining the step change of max_depth moving from one model to another
max_depth_starting_point, max_depth_ending_point = low_high_param\
(max_depth_midpoint,max_depth_step_size, 3) # finding lowest and highest values for depth_features of Gradientboosting
gradientboost_models = np.zeros((27,2)) # defining an array for accuracy scores
model_idx = 0     # index for each model                           
best_model_score = 0    # variable for the best score    
                ### training and testing(through validation sets) all the Gradientboosting models
for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for learning_rate_idx in range(learning_rate_starting_point, learning_rate_ending_point, learning_rate_step_size):      
        for max_depth_idx in range(max_depth_starting_point, max_depth_ending_point, max_depth_step_size):  
            os.system('cls')
            print("GradientBoosting Progress : ",int(progress_gradientboosting),"%")  # showing the progress of the process
            print("Number of estimator of {}     Learning rate of {} and Maximum depth of {} \
                    gives accuracy score of {:.2f}%".format(n_estimator_idx, learning_rate_idx/100, \
                    max_depth_idx, regr_score_gradientboosting*100))  # score of one single Gradientboosting model
            print("Best Score so far : ", round(best_regr_score_gradientboosting*100,2), "%")
            progress_gradientboosting += 100/27
            gradientboostingregressor = GradientBoostingRegressor(n_estimators = n_estimator_idx,  \
                learning_rate = learning_rate_idx/100, max_depth = max_depth_idx) # assigning an instance of a Gradientboosting
            gradientboostingregressor.fit(X_train, Y_train)  # fitting the train arrays to each Gradientboosting model
            Y_prediction = np.around(gradientboostingregressor.predict(X_validation)) #predicting using the validation set
            regr_score_gradientboosting = accuracy_score(Y_validation, Y_prediction) #accuaracy score for each model
            gradientboost_models[model_idx, 1] = regr_score_gradientboosting * 100 # saving the score for each model
            gradientboost_models[model_idx, 0] = model_idx       # saving the index for each model
            if regr_score_gradientboosting > best_regr_score_gradientboosting: # assessing if the new score is better 
                best_n_estimators_gradientboosting = n_estimator_idx      # than the previous best score
                best_learning_rate_gradientboosting = learning_rate_idx 
                best_max_depth_gradientboosting = max_depth_idx  
                best_regr_score_gradientboosting = regr_score_gradientboosting
                best_model_score = gradientboost_models[model_idx, 1]
                best_model_idx = gradientboost_models[model_idx, 0]
            model_idx+=1 
os.system('cls')
print("\n             GradientBoosting performance ")
print("            --------------------------------") # show the best model of Gradientboosting tested on validation set
print("Best Regressor Score for Random Forest : {:.2f}%".format(best_regr_score_gradientboosting*100)) 
print("Best Estimator number : ", best_n_estimators_gradientboosting, "\nBest Learning rate : "\
      , best_learning_rate_gradientboosting/100, "\nbest_max_depth : ",best_max_depth_gradientboosting)

plot_models(gradientboost_models[:,0], gradientboost_models[:,1], "GradientBoost-Models", 'r') # Plot all the models

print("Best model : model ",int(best_model_idx), " with score of ", round(best_model_score,2), "%" )

input("Press Enter to continue...")

###################################################################################################################
##################### This part is implementing StackingRegressor(Part 4 of the project) ##########################
###################################################################################################################

######### Assigning the best performed HyperParameters in the RandomForest, AdaBoost and GradientBoosting ##########
reg1 = RandomForestRegressor(n_estimators = best_n_estimators_randomforest\
    ,  max_features = best_max_features_randomforest, max_depth = best_max_depth_randomforest)
reg2 = AdaBoostRegressor(n_estimators = best_n_estimators_adaboost,  learning_rate = best_learning_rate_adaboost/100)
reg3 = GradientBoostingRegressor(n_estimators = best_n_estimators_gradientboosting\
    ,  learning_rate = best_learning_rate_gradientboosting/100, max_depth = best_max_depth_gradientboosting)

os.system('cls')
print("\n              StackingRegressor performance ")
print("            --------------------------------")
print("Please wait few moments for the final result...")
stackingregressor = StackingRegressor(estimators=[('rf', reg1), ('ad', reg2), ('gb', reg3)])
stackingregressor = stackingregressor.fit(X_train, Y_train)        ### Fitting the StackingRegressor model #####  
Y_pred = np.around(stackingregressor.predict(X_validation)) ##### Predicting the Labels based on the test features #####
ensemble_score = accuracy_score(Y_validation, Y_pred)  ##### Calulating the accuracy of each model ####

print("Regressor Score for StackingRegressor : {:.2f}%".format(ensemble_score*100))  ## showing the ensemble score ##
    
input("Press Enter to continue...")

###################################################################################################################
########## Using Linear Regression to represent the features impact on the labels(Part 6 of the project) ##########
###################################################################################################################
os.system('cls')
linearregression = LinearRegression()
linearregression.fit(X_train, Y_train) ### Fitting the LinearRegression model #####

Y_pred = np.around(linearregression.predict(X_validation))  ##### Predicting the Labels based on the test features #####
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
input("Press Enter to continue...")
######## Showing the list of features ranked based the the impact to the prediction ############
show_features_impact(linearregression.coef_, X_train, header)
input("Press Enter to continue...")
###################################################################################################################



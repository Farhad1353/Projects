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
from regression_library import low_high_param


### This is Part 1 of the project where we read the train and test csv files and splittig them to features and labels

path_train_file = "../Data/winequality-red-train.csv"  # assigning the train.csv file path to a variable
path_test_file = "../Data/winequality-red-test.csv"  # assigning the test.csv file path to a variable

my_train_File, my_test_File = read_csv_files(path_train_file, path_test_file)  # reading the csv files using pandas
                                                                       
header, X_train, X_validation, Y_train, Y_validation\
= split_train_validation(my_train_File, split_rate=0.25)  # splitting the train.csv to train and validation arrays

X_test, Y_test = split_features_labels(my_test_File) # splitting the test.csv file to features and label arrays
                                                                                    
###################################################################################################################
############################### This part is using Random Forest in Regression form ###############################
###################################################################################################################
best_regr_score_randomforest = 0
progress_randomforest = 0
regr_score_randomforest = 0

n_estimator_midpoint, max_features_midpoint, max_depth_midpoint = read_param("../Data/RandomForest_reg.csv")
print(n_estimator_midpoint)
n_estimator_step_size = 50
n_estimator_strarting_point, n_estimator_ending_point = low_high_param(n_estimator_midpoint, n_estimator_step_size)

max_features_step_size = 1 
max_features_starting_point, max_features_ending_point = low_high_param(max_features_midpoint,max_features_step_size)
    
max_depth_step_size = 2  
max_depth_starting_point, max_depth_ending_point = low_high_param(max_depth_midpoint,max_depth_step_size)
print(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size)
         
for n_estimator_idx in range(n_estimator_strarting_point, n_estimator_ending_point, n_estimator_step_size):             
    for max_features_idx in range(max_features_starting_point, max_features_ending_point, max_features_step_size):      
        for max_depth_idx in range(max_depth_starting_point, max_depth_ending_point, max_depth_step_size):  
            os.system('cls')
            print("Random Forest Progress : ",int(progress_randomforest),"%")
            print("Number of estimator of {} Maximum features of {} and Maximum depth of {} \
                    gives accuracy score of {:.2f}%".format(n_estimator_idx, max_features_idx, \
                    max_depth_idx, regr_score_randomforest*100))
            progress_randomforest += 100/27
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
################################# This part is using AdaBoost in Regression form ##################################
###################################################################################################################
best_regr_score_ad, progress, regr_score = 0, 0 ,0   ##############################################
est_str, est_end, est_stp = 25, 45, 5                ## Setting the HyperParameters for AdaBoost ##
learn_str, learn_end, learn_stp = 20, 40, 5          ##############################################

########### total number of AdaBoost models is calcualeted here ########################################
total_steps = int((est_end+est_stp-est_str-1)/est_stp)*int((learn_end+learn_stp-learn_str-1)/learn_stp)

for est_idx in range(est_str, est_end, est_stp):            #################################################
    for maxf_idx in range(learn_str, learn_end, learn_stp): ## Nested loops to try all the HyperParameters ##
        os.system('cls')                                    #################################################
        print("AdaBoost Progress : ",int(progress),"%")
        print("Number of estimator of {} and Learning rate of {:.2f} gives accuracy score of {:.2f}%".format(est_idx, maxf_idx/100, regr_score*100))
        progress += 100/total_steps
        regr = AdaBoostRegressor(n_estimators =est_idx,  learning_rate = maxf_idx/100)
        regr.fit(X_train, Y_train)   ### Fitting the AdaBoost model #####
        Y_pred = np.around(regr.predict(X_validation))  ##### Predicting the Labels based on the test features #####
        regr_score = accuracy_score(Y_validation, Y_pred)  ##### Calulating the accuracy of each model #####
        if regr_score > best_regr_score_ad:           ################################################
            best_n_estimators_ad = est_idx            ##  Assessing if the model has the best score ##
            best_learning_rate_ad = maxf_idx/100      ## so far based on the chosen HyperParameters ##
            best_regr_score_ad = regr_score           ################################################
os.system('cls')
print("\n               AdaBoost performance ")
print("            --------------------------------")
print("Best Regressor Score for AdaBoost : {:.2f}%".format(best_regr_score_ad*100)) ## showing the best score ##
################# showing the best estimaters #################
print("Best Estimator number : ", best_n_estimators_ad, "\nBest Learning rate : ", best_learning_rate_ad)

input("Press Enter to continue...")

###################################################################################################################
############################### This part is using GradientBoost in Regression form ###############################
###################################################################################################################
best_regr_score_gb, progress, regr_score = 0, 0, 0   #####################################
est_str, est_end, est_stp = 75, 90, 5                #### Setting the HyperParameters ####
learn_str, learn_end, learn_stp = 19, 22, 1          #### for GradientBoost Regressor ####
maxd_str, maxd_end, maxd_stp = 5, 7, 1               #####################################

########### total number of GradientBoost models is calcualeted here ########################################
total_steps = int((est_end+est_stp-est_str-1)/est_stp)*int((learn_end+learn_stp-learn_str-1)/learn_stp)*int((maxd_end+maxd_stp-maxd_str-1)/maxd_stp)

for est_idx in range(est_str, est_end, est_stp):              #################################################
    for maxf_idx in range(learn_str, learn_end, learn_stp):   ## Nested loops to try all the HyperParameters ##
        for maxd_idx in range(maxd_str, maxd_end, maxd_stp):  #################################################
            os.system('cls')
            print("GradientBoost Progress : ",int(progress),"%")
            print("Number of estimator of {}, Learning rate of {:.2f} and Maximum depth of {}  gives accuracy score of {:.2f}%".format(est_idx, maxf_idx/100, maxd_idx, regr_score*100))
            progress += 100/total_steps
            regr = GradientBoostingRegressor(n_estimators =est_idx,  learning_rate = maxf_idx/100, max_depth = maxd_idx)
            regr.fit(X_train, Y_train)   ### Fitting the GradientBoost model #####
            Y_pred = np.around(regr.predict(X_validation)) ##### Predicting the Labels based on the test features #####
            regr_score = accuracy_score(Y_validation, Y_pred)  ##### Calulating the accuracy of each model #####
            if regr_score > best_regr_score_gb:           ##########################################
                best_n_estimators_gb = est_idx            ##  Assessing if the model has the best ##
                best_learning_rate_gb = maxf_idx/100      ##  score so far based on the chosen    ##
                best_max_depth_gb = maxd_idx              ##           HyperParameters            ##                     ##
                best_regr_score_gb = regr_score           ##########################################
os.system('cls')
print("\n              GradientBoost performance ")
print("            --------------------------------")
print("Best Regressor Score for GradientBoost : {:.2f}%".format(best_regr_score_gb*100)) ## showing the best score ##
################# showing the best estimaters #################
print("Best Estimator number : ", best_n_estimators_gb, "\nBest Learning rate : ", best_learning_rate_gb, "\nbest_max_depth : ",best_max_depth_gb)

input("Press Enter to continue...")

###################################################################################################################
############################# This part is implementing VotingRegressor ###########################################
###################################################################################################################

########### Assigning the best performed HyperParameters to the RandomForest, AdaBoost and GradientBoost ##########
reg1 = RandomForestRegressor(n_estimators = best_n_estimators_randomforest,  max_features = best_max_features_randomforest, max_depth = best_max_depth_randomforest)
reg2 = AdaBoostRegressor(n_estimators = best_n_estimators_ad,  learning_rate = best_learning_rate_ad)
reg3 = GradientBoostingRegressor(n_estimators = best_n_estimators_gb,  learning_rate = best_learning_rate_gb, max_depth = best_max_depth_gb)

ereg_arr = np.empty(4, dtype=object)
regr_best_score = 0
os.system('cls')
print("\n              VotingRegressor performance ")
print("            --------------------------------")
ereg = VotingRegressor(estimators=[('rf', reg1), ('ad', reg3), ('gb', reg2)])
ereg = ereg.fit(X_train, Y_train)          ### Fitting the VotingRegressor model #####
Y_pred = np.around(ereg.predict(X_validation))   ##### Predicting the Labels based on the test features #####
regr_score = accuracy_score(Y_validation, Y_pred)   ##### Calulating the accuracy of each model #####

print("Regressor Score for VotingRegressor : {:.2f}%".format(regr_score*100))  ## showing each score ##

input("Press Enter to continue...")

###################################################################################################################
######################### This part is implementing StackingRegressor #############################################
###################################################################################################################

########### Assigning the best performed HyperParameters to the RandomForest, AdaBoost and GradientBoost ##########
reg1 = RandomForestRegressor(n_estimators = best_n_estimators_randomforest,  max_features = best_max_features_randomforest, max_depth = best_max_depth_randomforest)
reg2 = AdaBoostRegressor(n_estimators = best_n_estimators_ad,  learning_rate = best_learning_rate_ad)
reg3 = GradientBoostingRegressor(n_estimators = best_n_estimators_gb,  learning_rate = best_learning_rate_gb, max_depth = best_max_depth_gb)

os.system('cls')
print("\n              StackingRegressor performance ")
print("            --------------------------------")
ereg = StackingRegressor(estimators=[('rf', reg1), ('ad', reg2), ('gb', reg3)])
ereg = ereg.fit(X_train, Y_train)        ### Fitting the StackingRegressor model #####  
Y_pred = np.around(ereg.predict(X_validation)) ##### Predicting the Labels based on the test features #####
regr_score = accuracy_score(Y_validation, Y_pred)  ##### Calulating the accuracy of each model ####

print("Regressor Score for StackingRegressor : {:.2f}%".format(regr_score*100))  ## showing each score ##
    

input("Press Enter to continue...")

###################################################################################################################
######## PART 6This part is using Linear Regression to represent the most effective features impacting the labels #######
###################################################################################################################
os.system('cls')
regr = LinearRegression()
regr.fit(X_train, Y_train) ### Fitting the LinearRegression model #####
print("\n              Linear Regresson performance ")
print("            --------------------------------")

Y_pred = np.around(regr.predict(X_validation))  ##### Predicting the Labels based on the test features #####
regr_score = accuracy_score(Y_validation, Y_pred)  ##### Calulating the accuracy of the model #####
print("Regressor Score for LinearRegression: {:.2f}%".format(regr_score*100))
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
input("Press Enter to continue...")
######## Showing the list of features ranked based the the impact to the prediction ############
show_features_impact(regr.coef_, X_train, header)
input("Press Enter to continue...")

###################################################################################################################



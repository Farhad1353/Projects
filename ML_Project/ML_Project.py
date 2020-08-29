import sys
sys.path.append('..')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import rgb2gray, get_regression_data, visualise_regression_data

from regression_library import generate_new_list, L, convert_to_original, convert_to_normal, MultiVariableLinearHypothesis, train_test, normalize_data

myFile = pd.read_csv("../Data/winequality-red-real.csv", sep=';', header = None)
X=np.array(myFile)
#new_wine_list = np.array([[8,0.5,0.05,2,0.07,12,45,.997,3.2,0.5,9.5],[8,.89,0.04,1.75,0.055,4,9,.9973,3.38,0.35,8.9],[7.1,.795,0.01,2.45,0.065,7.2,21,.9972,3.49,0.55,9.5],[7.6,0.53,0.41,2.2,0.088,9,41,.9972,3.59,0.62,10.9],[5.5,0.84,0.09,1.1,0.045,14,94,0.995,3.56,.86,13],[11,.3,.46,6.2,0.074,6,14,.9975,3.2,0.82,12]])
header = X[0,:]     # separating the header
X=X[1:,:].astype(np.float)  #converting the rest of the data to float
n_models = 10  # number of models in my initial method
Y = X[:,-1]  # separating the labels
X = X[:,:-1]  # separating the features
num_in_list = 5  # number of new wines to estimate thier quality
new_wine_list = generate_new_list(X, num_in_list) # genarateing the features for each new one randomly
quality_total = np.zeros(new_wine_list.shape[0]) #creating an array to assign the total qualities through estimation
for k in range(n_models):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle = True) #splitting the train and test data
    Y_train_original = Y_train  # keeping the original labels to be able to convert the normal figures back to thier original
    X_train_original = X_train  # keeping the original features to be able to convert the normal figures back to thier original
    X_test = normalize_data(X_test)  # normalizing all the feautures and labels of the existing Data
    Y_test = normalize_data(Y_test)
    X_train = normalize_data(X_train)
    Y_train = normalize_data(Y_train)

    H = MultiVariableLinearHypothesis(n_features= len(header) - 1, regularisation_factor= 0.01)
  
    y_hat = H(X_train) # predicting the normalized figures prior to training
    dLdw = H.calc_deriv(X_train, y_hat, Y_train)
    #print(H.w)
    train_test(200, X_train, Y_train, X_test, Y_test, H, L, 0.01, plot_cost_curve=True) # train model and test the model at the same time

    total = np.zeros(new_wine_list.shape[0]) # creating an array to assign the normalized new wine features
   
    new_wine_normal = convert_to_normal(new_wine_list, X_train_original) #normalizing the new wine features

    total = np.zeros(new_wine_normal.shape[0])
    for idx in range(new_wine_normal.shape[0]): #predicting the qualities of the new wines in range of [0 to 1]
        for j in range(new_wine_normal.shape[1]):
            total[idx] += new_wine_normal[idx][j] * H.w[j]
    
    quality = convert_to_original(Y_train_original, total) # Converting the estimated qualities to the from the normalized figures
    quality_total += quality
    #
    # print("The quality of the new wines are : ", total, " & ", quality, " & ",np.round(quality,0).astype(np.int))
quality_average = quality_total/n_models    # average predicted qualities of all the models
print("The average quality of the new wines are : ", np.round(quality_average,0).astype(np.int), "\n" )

for feature_weight in range(len(header)-1):
    print("weight for ", header[feature_weight]," is {:.3f}".format(H.w[feature_weight]))

###################################################################################################################
### This part is using Random Forest,AdaBoost and GradientBoost in both forms of Classifier and Regressor##########
###################################################################################################################
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=500,  max_depth=10)
regr.fit(X, Y)
print("Regressor Score for Random Forest: {:.2f}".format(regr.score(X, Y)))
print(regr.predict(new_wine_list).astype(np.int))

from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier(max_depth=10, max_samples=200) # init random forest
randomForest.fit(X, Y) # fit random forest of decision trees
#print("Classifier Score for Random Forest: ",randomForest.score(X, Y))
#print(randomForest.predict(new_wine_list).astype(np.int))

from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor(random_state=20,  n_estimators=200, learning_rate=0.1, loss= 'linear')
regr.fit(X, Y)
print("Regressor Score for AdaBoost: {:.2f}".format(regr.score(X, Y)))
print(regr.predict(new_wine_list).astype(np.int))

from sklearn.ensemble import AdaBoostClassifier

Adaboost = AdaBoostClassifier(n_estimators=500)
Adaboost.fit(X, Y)
#print("Classifier Score for AdaBoost: ", Adaboost.score(X, Y))
#print(Adaboost.predict(new_wine_list).astype(np.int))

from sklearn.ensemble import GradientBoostingRegressor
regr = GradientBoostingRegressor(n_estimators=1000)
regr.fit(X, Y)
print("Regressor Score for GradientBoost: {:.2f}".format(regr.score(X, Y)))
print(regr.predict(new_wine_list).astype(np.int))

from sklearn.ensemble import GradientBoostingClassifier

Gradientboost = GradientBoostingClassifier(n_estimators=100)
Gradientboost.fit(X, Y)
#print("Classifier Score for GradientBoost: ",Gradientboost.score(X, Y))
#print(Gradientboost.predict(new_wine_list).astype(np.int))


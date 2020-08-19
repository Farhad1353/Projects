import numpy as np
import pandas as pd
from utils import rgb2gray, get_regression_data, visualise_regression_data
import matplotlib.pyplot as plt

from regression_library import L, convert_to_original, LinearHypothesis, MultiVariableLinearHypothesis, train, standardize_data, normalize_data

myFile = pd.read_csv('winequality-red-real.csv', sep=';', header = None)
X=np.array(myFile)

header = X[0,:]
X=X[1:,:].astype(np.float)

Y_ = X[:,-1]

X = normalize_data(X)

Y= X[:,-1]

X = X[:,:-1]

H = MultiVariableLinearHypothesis(n_features= len(header) - 1, regularisation_factor= 0.01)
    
    #plt.figure()
    #plt.scatter(X_sorted, Y_sorted, c='r', label='Label')
    #plt.legend()
    #plt.xlabel(header[feature])
    #plt.ylabel('Y')
    #plt.show()

y_hat = H(X)
dLdw, dLdb = H.calc_deriv(X, y_hat, Y)

train(200, X, Y, H, L, 0.01, plot_cost_curve=True) # train model and plot cost curve
#visualise_regression_data(X[:,feature], Y, H(X[:,feature])) # plot predictions and true data
total = 0
features_new_wine = np.array([8,1,1,2,0,12,45,1,5,14,8])
features_new_wine = normalize_data(features_new_wine)
for idx in range(len(features_new_wine)):
    total += features_new_wine[idx] * H.w[idx]

quality = convert_to_original(Y_, total)

print("The quality of the new wine is : ", np.round(quality,0).astype(np.int))






import numpy as np
import pandas as pd
from utils import rgb2gray, get_regression_data, visualise_regression_data
import matplotlib.pyplot as plt

from regression_library import L, LinearHypothesis, MultiVariableLinearHypothesis, train, standardize_data, normalize_data

myFile = pd.read_csv('winequality-red-real.csv', sep=';', header = None)
X=np.array(myFile)

header = X[0,:]
#print(header)

Y_f_half = np.zeros(len(header))
Y_l_half = np.zeros(len(header))
f_f_half = np.zeros(len(header))
f_l_half = np.zeros(len(header))

X=X[1:,:].astype(np.float)

X = normalize_data(X)

Y = X[:,-1]

X = X[:,:-1]

H = MultiVariableLinearHypothesis(n_features= len(header) - 1, regularisation_factor= 0.01)
H.b = 0

for feature in range(len(header)-1):
    sort_idxs = np.argsort(X[:,feature])
    X_sorted = X[:,feature][sort_idxs]
    Y_sorted = Y[sort_idxs]
    #print("Y_sorted :", Y_sorted[int(len(sort_idxs)/2):], Y_sorted[int(len(sort_idxs)/2):].shape)
    Y_f_half=np.mean(Y_sorted[0:int(len(sort_idxs)/2)])
    Y_l_half=np.mean(Y_sorted[int(len(sort_idxs)/2):])
    f_f_half=np.mean(X_sorted[0:int(len(sort_idxs)/2)])
    f_l_half=np.mean(X_sorted[int(len(sort_idxs)/2):])

    
    H.w[feature] = (Y_l_half - Y_f_half) / (f_l_half - f_f_half)
    H.b += np.median(Y_sorted) - np.median(X_sorted)
    print("Feature ", feature, " W :", "%.4f" % H.w[feature], "B : ", "%.1f" %H.b)

    H.b = np.random.randn()
    H.w = np.random.randn(len(header)-1)

    #plt.figure()
    #plt.scatter(X_sorted, Y_sorted, c='r', label='Label')
    #plt.legend()
    #plt.xlabel(header[feature])
    #plt.ylabel('Y')
    #plt.show()

y_hat = H(X)
dLdw, dLdb = H.calc_deriv(X, y_hat, Y)

train(100, X, Y, H, L, 0.01, plot_cost_curve=True) # train model and plot cost curve
#visualise_regression_data(X[:,feature], Y, H(X[:,feature])) # plot predictions and true data








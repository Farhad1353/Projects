import sys
sys.path.append('..')
import numpy as np
import pandas as pd 
import sklearn.datasets
import matplotlib.pyplot as plt
from get_colors import colors

def plot_models(X, Y, title_name):
    plt.figure()
    plt.scatter(X, Y, c='r')
    plt.title(title_name)
    plt.xlabel('Model number')
    plt.ylabel('Accuracy %')
    plt.show()


def low_high_param(mid, step, param=3):
    if param == 3:
        low = mid - step
        high = mid + step + 1
    else:
        low = mid - (step * 2)
        high = mid + (step * 2) + 1
    return low, high

def show_features_impact(r, X_train, header):
    convert_coef = np.zeros((len(r),2))

    for i in range(len(r)):
        convert_coef[i][0] = round(r[i] * (np.max(X_train[:,i])-np.min(X_train[:,i])),1)
        header[i] = np.char.capitalize(header[i])
        if convert_coef[i][0] > 0 :
            convert_coef[i][1] = +1
        else:
            convert_coef[i][1] = -1
            convert_coef[i][0] *= -1
    header_feature = np.expand_dims(header[:-1], axis=1)
    convert_coef_big = np.append(header_feature,convert_coef, axis=1)
    convert_coef_big = convert_coef_big[convert_coef_big[:,1].argsort()][::-1]
    print("\n            Rank Highest impacting features ")
    print("          --------------------------------------")
    for i in range(len(r)):
        pos_neg = "positive"
        if convert_coef_big[i,2] == -1:
            pos_neg = "negative"
        print("%-20s" %convert_coef_big[i,0], " with weight of  ",convert_coef_big[i,1]," and ",pos_neg,"imapct")





    




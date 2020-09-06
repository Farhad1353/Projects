import sys
sys.path.append('..')
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split

def read_param(path, param=3):
    my_file = pd.read_csv(path, header = None)
    my_param = np.array(my_file)
    if param == 3:
        param1 = np.squeeze(my_param[0].astype(np.int))
        param2 = np.squeeze(my_param[1].astype(np.int))
        param3 = np.squeeze(my_param[2].astype(np.int))
        return param1, param2, param3
    else:
        param1 = np.squeeze(my_param[0].astype(np.int))
        param2 = np.squeeze(my_param[1].astype(np.int))
        return param1, param2

def read_csv_files(path1, path2):
    train_File = pd.read_csv(path1, header = None) 
    test_File = pd.read_csv(path2, header = None)
    return(train_File, test_File)   

def split_train_validation(file_name, split_rate):
    X_train_val = np.array(file_name)                                                                                               ### Assigning     ###
    header = X_train_val[0,1:]                                                               
    X_train_val = X_train_val[1:,1:].astype(np.float)  
    Y_train_val = X_train_val[:,-1] 
    X_train_val = X_train_val[:,:-1]  
    X_train, X_val, Y_train, Y_val \
    = train_test_split(X_train_val, Y_train_val, test_size=split_rate, random_state=42)
    return(header, X_train, X_val, Y_train, Y_val)

def split_features_labels(file_name):
    X_ = np.array(file_name)                                                            
    X_ = X_[1:,1:].astype(np.float)  
    Y_ = X_[:,-1] 
    X_ = X_[:,:-1]  
    return(X_, Y_)




    




import sys
sys.path.append('..')
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split

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




    




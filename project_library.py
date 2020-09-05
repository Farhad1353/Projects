import sys
sys.path.append('..')
import numpy as np
import pandas as pd 

def read_csv_files(path1, path2):
    train_File = pd.read_csv(path1, header = None) 
    test_File = pd.read_csv(path2, header = None)
    return(train_File, test_File)   
                                                                              
#X_train_validation = np.array(my_train_File)                                                                                               ### Assigning     ###
#header = X_train_validation[0,1:]                                                               
#X_train_validation = X_train_validation[1:,1:].astype(np.float)  
#Y_train_validation = X_train_validation[:,-1] 
#X_train_validation = X_train_validation[:,:-1]  

#X_train, X_validation, Y_train, Y_validation \
#= train_test_split(X_train_validation, Y_train_validation, test_size=0.25, random_state=42)




    




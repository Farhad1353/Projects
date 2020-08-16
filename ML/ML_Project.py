import sys
sys.path.append('..')

import numpy as np
#from utils import rgb2gray

def gs_ortho(vector_list, column_idx=0): # takes as input the list of column vectors
    n_rows, n_columns = np.shape(vector_list)
    
    # Base case:
    if column_idx == n_columns:
        return vector_list##
    
    # Recursion call: # use loop to create a projection list, then subtract by the sum of its components
    else:
        current_v = vector_list[:, column_idx]##
        projection_list = []
        for i in range(column_idx):
            e_i = vector_list[:, i]
            inner = np.dot(current_v, e_i)##
            projection = inner*e_i
            projection_list.append(projection) ##
        
        projection_list = np.array(projection_list).T
        current_u = current_v - np.sum(projection_list, axis=-1)
        current_e = current_u/np.linalg.norm(current_u)
        
        vector_list[:, column_idx] = current_e ##
        
        return gs_ortho(vector_list, column_idx=column_idx+1)##
    
A = np.array([[1, 2, 2], [2, 1, 2], [2, 2, 1]], dtype="float_")
print("Q =" ,gs_ortho(A))

# Defining our matrix
A = np.array([[1,2,2],[2,1,2],[2,2,1]])

# Applying Orthonormalization (first output of the following function)
Q,_ = np.linalg.qr(A)
print("Q =", Q)

def reorder_eig_vectors(eig_tuple):
    eig_values, eig_vectors = eig_tuple
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    return eig_values, eig_vectors

  # Define our original matrix
A = np.array([[1, 3, 2],
              [2 , 2, 1]])

def preU_matrix(A):
    return np.matmul(A, A.T)

AA_T = preU_matrix(A)
print("AA_T:")
print(AA_T)

def U_matrix(A):
    AA_T = preU_matrix(A)
    U_ = np.linalg.eig(AA_T)
    U_eigvalues, U_eigvectors = reorder_eig_vectors(U_)
    U,_ = np.linalg.qr(U_eigvectors)
    return U_eigvalues, U

U_eigvalues, U = U_matrix(A)
print("Eigenvalues:", U_eigvalues)
print("U:")
print(U)

# CODE BREAK CHALLENGE:

# Computing the augmented matrix
def preV_matrix(A):
    return np.matmul(A.T, A)

# Computing V matrix
def V_matrix(A):
    A_TA = preV_matrix(A)
    V_ = np.linalg.eig(A_TA)
    V_eigvalues, V_eigvectors = reorder_eig_vectors(V_)
    V,_ = np.linalg.qr(V_eigvectors)
    return V_eigvalues, V

V_eigvalues, V = V_matrix(A)
V_T = V.T
print("Eigenvalues:", V_eigvalues)
print("V:")
print(V)
print("V_T")
print(V_T)

# Constructing our Sigma matrix
def S_matrix(A):
    rows, columns = np.shape(A)
    U_ = U_matrix(A)
    
    # Getting only appropriate number of eigenvalues in correct order
    eigvalues,_ = reorder_eig_vectors(U_)
    min_dim = min(rows, columns)
    eigvalues = eigvalues[0:min_dim]
    
    S = np.zeros((rows,columns))
    np.fill_diagonal(S,np.sqrt(eigvalues))
    return S

S = S_matrix(A)
print("Sigma:")
print(S)
print()

# Checking whether we get the original matrix (Numpy function allclose to check if it is the same as the original)
A_check = np.matmul(U, np.matmul(S, V_T))
print("A_reconstruct:")
print(A_check)
print()
print(np.allclose(A, A_check))

def svd(A):
    U,_ = U_matrix(A)
    S = S_matrix(A)
    V,_ = V_matrix(A)
    return U, S, V.T

U, S, V_T = svd(A)
print("U:")
print(U)
print("Sigma:")
print(S)
print("V_T :")
print(V_T)
print()
U, S, V_T = np.linalg.svd(A)
print("U:")
print(U)
print("Sigma:")
print(S)
print("V_T:")
print(V_T)



# Uses SVD to construct matrix approximations
def reduced_svd(A, R):
    U, S, V_T = np.linalg.svd(A)
    
    # Obtaining approximation of matrices with R components
    U_hat = U[:,0:R]
    S_hat = S[0:R]
    V_T_hat = V_T[0:R,:]
    return U_hat, S_hat, V_T_hat

# Computes apprimation of original matrix
def approx(A, R):
    U_hat, S_hat, V_T_hat = reduced_svd(A, R)
    S_hat_matrix = np.zeros((S_hat.shape[0], S_hat.shape[0]))
    np.fill_diagonal(S_hat_matrix, S_hat)

    return np.matmul(U_hat, np.matmul(S_hat_matrix, V_T_hat))

 # Displaying approximation

import matplotlib.pyplot as plt

# Importing our image
img = plt.imread("images/svd_skeleton.jpg")
#img = rgb2gray(img)

# Determining parameters
R_vals = [5, 10, 50, 100, 500]

# Displaying approximation
for R in R_vals:
    new_approx = approx(img, R)
    print("R =", R)
    plt.figure()
    plt.imshow(new_approx, cmap='gray')
    plt.show()
   
import numpy as np
import matplotlib.pyplot as plt 
from Xavier import Xavier_normal, Xavier_uniform

# DEFINE MEAN SQUARED ERROR LOSS FUNCTION
def L(y_hat, labels):
    errors = y_hat - labels # calculate errors
    squared_errors = np.square(errors) # square errors
    mean_squared_error = np.sum(squared_errors) / (len(y_hat)) # calculate mean 
    return mean_squared_error # return loss

def generate_new_list(X, num):   # generating the features of our new wines. Based on mean and STD of the existing Data
    new_list = np.zeros((num, X.shape[1]))
    mean_std_list = np.zeros((X.shape[1], 2))
    for i in range(X.shape[1]):
        mean_std_list[i][0] = np.mean(X[:,i])
        mean_std_list[i][1] = np.std(X[:,i])
    for j in range(num):
        for i in range(X.shape[1]):
            new_list[j][i] = np.random.normal(mean_std_list[i][0], mean_std_list[i][1])
            if new_list[j][i] < 0:
                new_list[j][i] = 0

    np.set_printoptions(formatter={'float': "{0:0.3f}".format})
    #print(mean_std_list)
    #print(new_list)
    #wait = input("PRESS ENTER TO CONTINUE.")
    return new_list
    



class MultiVariableLinearHypothesis:
    def __init__(self, n_features, regularisation_factor): ## add regularisation factor as parameter
        self.n_features = n_features
        self.regularisation_factor = regularisation_factor ## add self.regularisation factor
        self.w = np.random.rand(n_features)
                
    def __call__(self, X): # what happens when we call our model, input is of shape (n_examples, n_features)
        y_hat = np.matmul(X, self.w) # make prediction, now using vector of weights rather than a single value
        return y_hat # output is of shape (n_examples, 1)
    
    def update_params(self, new_w):
        self.w = new_w
        
    def calc_deriv(self, X, y_hat, labels):
        m = len(labels)
        diffs = y_hat-labels
        dLdw = dLdw = 2 / m * np.matmul(X.T, diffs)
        dLdw += 2 * self.regularisation_factor * self.w ## add regularisation term gradient
        return dLdw


def normalize_data(dataset):

    min_data, max_data = np.min(dataset, axis=0), np.max(dataset, axis=0) ## get min and max of dataset
    normalized_dataset  = (dataset-min_data)/(max_data - min_data)
    return normalized_dataset

def convert_to_original(dataset, quality):
    conv_origin = np.zeros(len(quality))
    for i in range(len(quality)):
        conv_origin[i]  = quality[i] * (np.max(dataset) - np.min(dataset)) + np.min(dataset)
  
    return conv_origin

def convert_to_normal(new_example, X_train):
    converted = np.zeros((new_example.shape[0],X_train.shape[1]))
    for i in range(X_train.shape[1]):
        for j in range(new_example.shape[0]):
            converted[j][i] = (new_example[j][i] - np.min(X_train[:,i]))/(np.max(X_train[:,i]) - np.min(X_train[:,i]))
    return converted

def plot_loss(losses):
    plt.figure() # make a figure
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses) # plot costs

def train_test(num_epochs, X_tr, Y_tr,X_te, Y_te, H, L, learning_rate, plot_cost_curve=True):
    #print("Cost prior to training : ","%.2f" % L(H(X_tr), Y_tr))
    #print("Cost prior to testing : ","%.2f" % L(H(X_te), Y_te))
    all_costs_train = [] ## initialise empty list of costs to plot later
    all_costs_test = []
    for e in range(num_epochs): ## for this many complete runs through the dataset
        y_hat = H(X_tr) ## make predictions
        #print(y_hat.shape,Y.shape,"y_hat and Y")
        cost_train =L(y_hat, Y_tr)
        y_hat_test = H(X_te)
        cost_test = L(y_hat_test, Y_te) ## compute loss
        #print(cost_train, cost_test) 
        dLdw = H.calc_deriv(X_tr, y_hat, Y_tr) ## calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw ## compute new model weight using gradient descent update rule
        H.update_params(new_w) ## update model weight and bias
        all_costs_train.append(cost_train) ## add cost for this batch of examples to the list of costs (for plotting)
        all_costs_test.append(cost_test)
    cost_idx = np.linspace(1, num_epochs, num_epochs)
    #plt.figure()
    #plt.scatter(cost_idx, all_costs_train, c='r', label='Label')
    #plt.legend()
    #plt.xlabel('Epoch')
    #plt.ylabel('Cost')
    #plt.show()
    #if plot_cost_curve: ## plot stuff
    #    plot_loss(all_costs_train)
        
    #print('Final train cost:', "%.2f" % cost_train)

    #plt.figure()
    #plt.scatter(cost_idx, all_costs_test, c='r', label='Label')
    #plt.legend()
    #plt.xlabel('Epoch')
    #plt.ylabel('Cost')
    #plt.show()
    #if plot_cost_curve: ## plot stuff
    #    plot_loss(all_costs_test)
        
    #print('Final test cost:', "%.2f" % cost_test)
    #for feature in range(X.shape[1]):
     #   print('Weight value for feature', feature, "%.4f"% H.w[feature])
    #print('Bias value:', "%.1f" % H.b)





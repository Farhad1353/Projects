import numpy as np
import matplotlib.pyplot as plt 
from Xavier import Xavier_normal, Xavier_uniform

# DEFINE MEAN SQUARED ERROR LOSS FUNCTION
def L(y_hat, labels):
    errors = y_hat - labels # calculate errors
    squared_errors = np.square(errors) # square errors
    mean_squared_error = np.sum(squared_errors) / (len(y_hat)) # calculate mean 
    return mean_squared_error # return loss

class MultiVariableLinearHypothesis:
    def __init__(self, n_features, regularisation_factor): ## add regularisation factor as parameter
        self.n_features = n_features
        self.regularisation_factor = regularisation_factor ## add self.regularisation factor
        self.b = 0
        self.w = Xavier_normal(n_features)
        
    def __call__(self, X): # what happens when we call our model, input is of shape (n_examples, n_features)
        y_hat = np.matmul(X, self.w) + self.b # make prediction, now using vector of weights rather than a single value
        return y_hat # output is of shape (n_examples, 1)
    
    def update_params(self, new_w, new_b):
        self.w = new_w
        self.b = new_b
        
    def calc_deriv(self, X, y_hat, labels):
        m = len(labels)
        diffs = y_hat-labels
        dLdw = 2 * np.array([np.sum(diffs * X[:, i]) / m for i in range(self.n_features)]) 
        dLdw += 2 * self.regularisation_factor * self.w ## add regularisation term gradient
        dLdb = 0 * 2 * np.sum(diffs) / m
        return dLdw, dLdb

class LinearHypothesis:
    def __init__(self): 
        self.w = np.random.randn() ## weight
        self.b = np.random.randn() ## bias
    
    def __call__(self, X): ## how do we calculate output from an input in our model?
        y_hat = self.w*X + self.b ## make linear prediction
        return y_hat
    
    def update_params(self, new_w, new_b):
        self.w = new_w ## set this instance's w to the new w
        self.b = new_b ## set this instance's b to the new b
        
    def calc_deriv(self, X, y_hat, labels):
        m = len(labels) ## m = number of examples
        diffs = y_hat - labels ## calculate errors
        dLdw = 2*np.array(np.sum(diffs*X) / m) ## calculate derivative of loss with respect to weights
        dLdb = 2*np.array(np.sum(diffs)/m) ## calculate derivative of loss with respect to bias
        return dLdw, dLdb ## return rate of change of loss wrt w and wrt b
    
 ## calculate gradient of current loss with respect to model parameters
def standardize_data(dataset):

    mean, std = np.mean(dataset, axis=0), np.std(dataset, axis=0) ## get mean and standard deviation of dataset
    standardized_dataset  = (dataset-mean)/std

    return standardized_dataset


def normalize_data(dataset):

    min_data, max_data = np.min(dataset, axis=0), np.max(dataset, axis=0) ## get mean and standard deviation of dataset
    normalized_dataset  = (dataset-min_data)/(max_data - min_data)
  
    return normalized_dataset

def convert_to_original(dataset, quality):

    conv_origin  = quality * (np.max(dataset, axis=0) - np.min(dataset, axis=0)) + np.min(dataset, axis=0)
  
    return conv_origin


def plot_loss(losses):
    plt.figure() # make a figure
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses) # plot costs

def train(num_epochs, X, Y, H, L, learning_rate, plot_cost_curve=True):
    print("Cost prior to training : ","%.2f" % L(H(X), Y))
    all_costs = [] ## initialise empty list of costs to plot later
    for e in range(num_epochs): ## for this many complete runs through the dataset
        y_hat = H(X) ## make predictions
        #print(y_hat.shape,Y.shape,"y_hat and Y")
        cost = L(y_hat, Y) ## compute loss 
        dLdw, dLdb = H.calc_deriv(X, y_hat, Y) ## calculate gradient of current loss with respect to model parameters
        new_w = H.w - learning_rate * dLdw ## compute new model weight using gradient descent update rule
        new_b = H.b - learning_rate * dLdb ## compute new model bias using gradient descent update rule
        H.update_params(new_w, new_b) ## update model weight and bias
        all_costs.append(cost) ## add cost for this batch of examples to the list of costs (for plotting)
    cost_idx = np.linspace(1, num_epochs, num_epochs)
    plt.figure()
    plt.scatter(cost_idx, all_costs, c='r', label='Label')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.show()
    if plot_cost_curve: ## plot stuff
        plot_loss(all_costs)
        
    print('Final cost:', "%.2f" % cost)
    #for feature in range(X.shape[1]):
     #   print('Weight value for feature', feature, "%.4f"% H.w[feature])
    #print('Bias value:', "%.1f" % H.b)




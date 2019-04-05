# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
import numpy as np

def computeCost(X, y, theta):
#COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y

# Initialize some useful values
    m = np.size(y, axis=0) # number of training examples

# You need to return the following variables correctly 

    h = np.dot(X, theta) - y
    J = np.sum( h*h)/(2*m)
    return J
    
    







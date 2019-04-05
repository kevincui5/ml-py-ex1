# -*- coding: utf-8 -*-

# Exercise 1: Linear Regression



# X refers to the population size in 10,000s
# y refers to the profit in $10,000s

import numpy as np
import warmUpExercise
import pandas as pd
#import plotData
import matplotlib.pyplot as plt

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
warmUpExercise.warmUpExercise()

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv('ex1data1.txt', header=None)
X = data.iloc[:, 0].values[:, None]
y = data.iloc[:, 1].values[:, None]
m = np.size(y, axis=0) 
plt.plot(X, y, 'rx')


# =================== Part 3: Cost and Gradient descent ===================
from computeCost import computeCost
X_nobias = X # for later use
X = np.hstack((np.ones((X.shape[0], 1)), X)) # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0  0]\nCost computed = %f\n', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
theta1 = np.array([[-1], [2]]) # 2x1
J = computeCost(X, y, theta1)
print('\nWith theta = [-1  2]\nCost computed = %f\n', J)
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
from gradientDescent import gradientDescent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('%f\n', theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')
'''
# Plot the linear fit
hold on # keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off # don't overlay any more plots on this figure
'''
# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([(1, 3.5)]), theta)
print('For population = 35,000, we predict a profit of %f\n',predict1*10000)
predict2 = np.dot(np.array([(1, 7)]), theta)
print('For population = 70,000, we predict a profit of %f\n',predict2*10000)
'''
print('Program paused. Press enter to continue.\n')
pause

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals))

# Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i) theta1_vals(j)]
	  J_vals(i,j) = computeCost(X, y, t)
    



# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals'
# Surface plot
figure
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0') ylabel('\theta_1')

# Contour plot
figure
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0') ylabel('\theta_1')
hold on
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2)
'''

from sklearn import datasets, linear_model, metrics 
 
X = X_nobias
# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                    random_state=1) 
# create linear regression object 
reg = linear_model.LinearRegression() 
  
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(X_test, y_test))) 
  
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 
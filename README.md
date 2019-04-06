# ml-py-ex1
The problem come from Andrew Ng's machine learning course projects from Coursera, and 
I'd like to implement them in python instead of matlab/octave
We are asked to implement linear regression with one
variable or feature given, the population of the cities, to predict profits for
 a food truck in that city. We are given the file ex1data1.txt contains the 
 dataset for our problem. The first column is the population of a city and the second column is
the profit of a food truck in that city. A negative value for profit indicates a
loss.
After the dataset from ex1data1.txt is imported, the data from first column is 
stored in X, designated as input matrix, and second column in y, as label vector.
The we try to fit our linear regression model with this dataset using gradient
decent.
Then we implement the cost function using "vectorized" method, and we use 
gradient descent to get the minimum value of the cost function.  Of courese our
final goal is not this value, but the matrix of parameter that let us reach this 
optimization objective.
Then we use the obtained parameters to make two predictions, the predicted profits
at two cities with certain populations.
Then we use linear regression model from sklearn library to perform the above process.

DO NOT USE MY SOURCE CODE TO COMPLETE THE EXERCISES/PROJECTS ON COURSERA MACHINE
 LEARNING COURSE.
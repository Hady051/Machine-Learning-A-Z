#### Importing the libraries
##"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

#@"""



#### Importing the Dataset
##"""

dataset = pd.read_csv('./datasets/Salary_Data.csv')
# print(dataset)
input_feature = dataset.iloc[ : , :-1].values
dependent_variable = dataset.iloc[ : , -1].values
# print(input_feature)
# print(dependent_variable)

#@"""



#### Splitting the dataset into train, test
##"""

from sklearn.model_selection import train_test_split

(input_feature_train, input_feature_test,
 dependent_variable_train, dependent_variable_test) = train_test_split(input_feature, dependent_variable,
                                                                       test_size=0.2, random_state=0)

#@"""



#### Training the Simple Linear Regression model
##"""

from sklearn.linear_model import LinearRegression

linear_R = LinearRegression()

linear_R.fit(input_feature_train, dependent_variable_train)

#@"""



#### Predicting Test set results
##"""

test_set_predicted_values = linear_R.predict(input_feature_test)

#@"""



#### Visualizing Training set results
"""

plt.scatter(input_feature_train, dependent_variable_train, c='orange')

train_set_predicted_values = linear_R.predict(input_feature_train)
plt.plot(input_feature_train, train_set_predicted_values, color='blue')
plt.title("Salary and Experience for Train Set")
plt.xlabel("Years of Experience")
plt.ylabel('Salary')
plt.show()

"""



#### Visualizing Test set results
##"""

plt.scatter(input_feature_test, dependent_variable_test, color='orange')

## SLR for training set
plt.plot(input_feature_train, linear_R.predict(input_feature_train), c='blue') # // like the video
# // according to the video, the SLR has a unique equation that the prediction for the train set is very similar
# // to the test set, so you don't necessarily have to change it

## SLR for test set
plt.plot(input_feature_test, test_set_predicted_values, c='red')
plt.title("Salary and Experience for Test Set with SLR prediction for both lines")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#@"""



#### predicting a single value
##"""

print(linear_R.predict([[12] ] ) )  # -> [138531.00067138]
# // meaning if you have 12 years of exp according to the model how much relatively
# // you're going to get as a salary.
# // **notice that (12) isn't a value in the plot, meaning that SLR predicts values not even in the curve
# // (predicting continuous values) which is basically what (Regression) is

# //Important note:
# // Notice that the value of the feature (12 years) was input in a double pair of square brackets.
# // That's because the "predict" method always expects a 2D array as the format of its inputs.
# // And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:
# // 12→scalar
#
# // [12]→1D array
#
# // [[12]]→2D array -> like a (matrix)

print(linear_R.predict([[8] ] ) )  # -> [101280.70016446]

#@"""



#### Getting the final linear regression equation with the values of the coefficients
##"""

# // the equation: y = b1x1 + bo, to get b0 and b1 of the final result after the training ended:

## b0: y-intercept
print("The final b0: ", linear_R.intercept_)  # -> The final b0: 26780.09915062818

## b1: slope
print("The final b1: ", linear_R.coef_)  # -> The final b1:  [9312.57512673]


#@"""

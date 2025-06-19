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

dataset = pd.read_csv('./Datasets/50_Startups.csv')
# print(dataset)
input_features = dataset.iloc[ : , :-1].values
dependent_variable = dataset.iloc[ : , -1].values
# print("input features: ", input_features)
# print("output feature: ", dependent_variable)

#@"""



#### Encoding Categorical Data
##"""

## Encoding the Independent variables
##'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# // to check for how many unique values, if the number is sufficiently low and depending on
# / the column's type, it can be considered a categorical feature
#!! print(dataset.nunique() )
# // state: 3, so state here is a categorical variable.

column_transformer = ColumnTransformer(transformers=[("encoding", OneHotEncoder(drop="first"), [3] ) ],  # // ** Q&A
                                       remainder='passthrough')
input_features = np.array(column_transformer.fit_transform(input_features) )
#!! print(input_features)

# // New York-> 0.0 0.0 1.0, California-> 1.0 0.0 0.0, Florida-> 0.0 1.0 0.0

# // ** after dropping 1 Dummy Variable from the 3
# // New York-> 0.0 1.0, California-> 0.0 0.0, Florida-> 1.0 0.0

# // ***
# // from Q&A, we better delete one of the Dummy Variables manually, (wasn't done in the course).
# // we have 3 dummy variables here from the OHE operation: Florida, California, NY
# // drop one of the dummy variables from OneHotEncoding(), either using drop='first' parameter
# / or slicing the column we want to drop
# // (note: we will end up with totally different coefficients, and the predictions will vary due to this)
# // dropping the intercept by adding the parameter fit_intercept=False to LinearRegression()


# // **
# // We don't have to apply feature scaling,
# / because the coefficients (b1, b2, etc.) will compensate to put everything on the same scale
# / so having some features very higher than others doesn't matter here.

#@"""



#### Splitting the data
##"""

from sklearn.model_selection import train_test_split

# // X_train, X_test, y_train, y_test, (tuple unpacking)
(input_features_train, input_features_test,
 dependent_variable_train, dependent_variable_test) = train_test_split(input_features, dependent_variable,
                                                                       test_size=0.2, random_state=0)

# // ** We don't have to apply feature scaling,
# // because the coefficients (b1, b2, etc.) will compensate to put everything on the same scale
# / so having some features very higher than others doesn't matter here.

#@"""



#### Training the Multiple Linear Regression Model on the Training set
##"""

from sklearn.linear_model import LinearRegression

## building the model
MLR = LinearRegression()
## training the model
MLR.fit(input_features_train, dependent_variable_train)

#@"""



#### Predicting the test set results
##"""

test_set_predicted_values = MLR.predict(input_features_test)

np.set_printoptions(precision=2)  # // to make values only two decimal points

print(np.concatenate((test_set_predicted_values.reshape(len(test_set_predicted_values), 1),
                    dependent_variable_test.reshape(len(dependent_variable_test), 1) ),
                     axis=1) )

## Output
'''
[[103015.2  103282.38]
 [132582.28 144259.4 ]
 [132447.74 146121.95]
 [ 71976.1   77798.83]
 [178537.48 191050.39]
 [116161.24 105008.31]
 [ 67851.69  81229.06]
 [ 98791.73  97483.56]
 [113969.44 110352.25]
 [167921.07 166187.94]]
'''

# // to concatenate(add) two arrays of
# / the test_set_predicted_values and the real values (dependent_variable_train)
# // the np.concatenate function requires the arrays in a tuple (with parenthesis)
# // I reshaped the two arrays into an array col shape (rows:len(array), cols:1)
# // to view them in a col. way in the end make the axis arg=1,
# / if axis=0 they will be viewed horizontally, in a row shape

#@"""



#### Making a single prediction
##"""

# // (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000,
#                                               Marketing Spend = 300000 and State = 'California')
# // California-> 1.0 0.0 0.0

#!! print("MLR model's prediction for your input is: ", MLR.predict([[1, 0, 0, 160000, 130000, 300000]] ) )
# -> MLR model's prediction for your input is: [181566.92]

# // **after dropping 1 Dummy Variable
print("MLR model's prediction for your input is: ", MLR.predict([[0, 0, 160000, 130000, 300000]] ) )
# -> MLR model's prediction for your input is: [181566.92]

#@"""



#### Getting the final linear regression equation with the values of the coefficients
##"""

## b0: y-intercept
print("The final b0: ", MLR.intercept_)  # // -> The final b0: 42467.52924853204

## bn: slopes/ coefficients (b1, b2, etc.)
print("The final bn: ", MLR.coef_)
# // -> The final bn:  [ 8.66e+01 -8.73e+02  7.86e+02  7.73e-01  3.29e-02  3.66e-02]
# // b1, b2, b3, b4, b5, b6.  (b1, b2, b3) are the dummy variables cols (D1, D2, D3)
# // from the "state" col. turning into 3 dummy var. cols.

# // ***
# // If I did the drop="first" of the Q&A, each dummy variable will have 2 encodings instead of 3

#@"""



#### NOTES
##"""

# // Dropping a dummy variable changed the coefficients (b1, b2, b3, etc.),
# // but didn't change the prediction results.
# // In the course, he says that SKlearn takes care of dropping the dummy variable,
# // but I'm not sure, whether it's a case of no difference because of the dataset or whether
# // sklearn actually takes care of it internally.

#@"""
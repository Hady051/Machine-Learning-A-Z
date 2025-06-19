#### Importing libraries
##"""

import numpy as np
import pandas as pd

#@"""



#### Importing dataset
##"""

dataset = pd.read_csv('datasets/Data.csv')
matrix_of_features = dataset.iloc[ : , :-1].values      # // X
dependent_variable = dataset.iloc[ : , -1].values  # // y

#@"""



#### Taking care of missing data
##"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(matrix_of_features[ : , 1:3])  # // the cols with integer values
matrix_of_features[ : , 1:3] = imputer.transform(matrix_of_features[ : , 1:3])

# print("the new matrix of features: ", matrix_of_features)

#@"""



#### Encoding Categorical Data
##"""

## Encoding the Independent variables
##'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print(dataset.nunique() ) # // to check for how many unique values, if the number is sufficiently low and depending on
# the column's type, it can be considered a categorical feature

column_transformer = ColumnTransformer(transformers=[("encoding", OneHotEncoder(), [0])]
                                       , remainder='passthrough')
matrix_of_features = np.array(column_transformer.fit_transform(matrix_of_features) )

# print(matrix_of_features)

#@'''

## Encoding the Dependent variable
##'''
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
dependent_variable = label_encoder.fit_transform(dependent_variable)

# print(dependent_variable)

#@'''

#@"""



#### Splitting the data
##"""

from sklearn.model_selection import train_test_split

# // X_train, X_test, y_train, y_test, (tuple unpacking)
(matrix_of_features_train, matrix_of_features_test,
 dependent_variable_train, dependent_variable_test) = train_test_split(matrix_of_features, dependent_variable,
                                                                       test_size=0.2, random_state=1)

print("matrix_of_features_train = \n", matrix_of_features_train)
print("matrix_of_features_test = \n", matrix_of_features_test)
print("dependent_variable_train = \n", dependent_variable_train)
print("dependent_variable_test = \n", dependent_variable_test)

#@"""
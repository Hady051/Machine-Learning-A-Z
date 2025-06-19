"""
Coding Exercise 3: Encoding Categorical Data for Machine Learning
1: Import required libraries - Pandas, Numpy, and required classes for this task - ColumnTransformer, OneHotEncoder, LabelEncoder.

2: Start by loading the Titanic dataset into a pandas data frame. This can be done using the pd.read_csv function. The dataset's name is 'titanic.csv'.

3: Identify the categorical features in your dataset that need to be encoded. You can store these feature names in a list for easy access later.

4: To apply OneHotEncoding to these categorical features, create an instance of the ColumnTransformer class. Make sure to pass the OneHotEncoder() as an argument along with the list of categorical features.

5: Use the fit_transform method on the instance of ColumnTransformer to apply the OneHotEncoding.

6: The output of the fit_transform method should be converted into a NumPy array for further use.

7: The 'Survived' column in your dataset is the dependent variable. This is a binary categorical variable that should be encoded using LabelEncoder.

8.  Print the updated matrix of features and the dependent variable vector

"""

#### Importing libraries
##"""

import numpy as np
import pandas as pd

#@"""



#### Importing dataset
##"""

dataset = pd.read_csv('datasets/titanic2.csv')
matrix_of_features = dataset.iloc[ : , 2: ].values      # // X
dependent_variable = dataset["Survived"]  # // y

# print(matrix_of_features)
# print(dataset)
# print(dataset.columns)

# print(dependent_variable)

#@"""



#### Taking care of missing data
"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(matrix_of_features[ : , 6])  # // the cols with integer values
matrix_of_features[ : , 6] = imputer.transform(matrix_of_features[ : , 6])

print("the new matrix of features: ", matrix_of_features)

#@"""



#### Encoding Categorical Data
##"""

## Encoding the Independent variables
##'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print(dataset.nunique() )
# categorical_features_cols = ['Sex', 'Embarked', 'Pclass']

# column_transformer = ColumnTransformer(transformers=[('encoding', OneHotEncoder(), categorical_features_cols)]  # ** no square brackets since its already a list
#                                        , remainder='passthrough')
# dataset = np.array(column_transformer.fit_transform(dataset) )   # ** dataset not matrix_of_features

# print(dataset[1:5])  # // showing the first 5 rows of the dataset numpy matrix
print(matrix_of_features[0])
# // Another way
column_transformer = ColumnTransformer(transformers=[('encoding', OneHotEncoder(), [0, 3, 10])] # cat. features indices according to matrix of features
                                       , remainder='passthrough')

matrix_of_features = np.array(column_transformer.fit_transform(matrix_of_features) )

print(matrix_of_features[0]) # // showing the first 5 rows of the matrix of features numpy matrix

#@'''

## Encoding the Dependent variable
##'''
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
dependent_variable = label_encoder.fit_transform(dependent_variable)

print(dependent_variable)

#@'''

#@"""


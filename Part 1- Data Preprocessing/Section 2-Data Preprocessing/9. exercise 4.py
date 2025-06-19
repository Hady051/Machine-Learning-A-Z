"""
Coding Exercise 4: Dataset Splitting and Feature Scaling
1: Import necessary Python libraries: pandas, train_test_split from sklearn.model_selection,
and StandardScaler from sklearn.preprocessing.

2: Load the Iris dataset using Pandas read.csv. Dataset name is iris.csv.

3: Use train_test_split to split the dataset into an 80-20 training-test set.

4: Apply random_state with 42 value in train_test_split function for reproducible results.

5: Print X_train, X_test, Y_train, and Y_test to understand the dataset split.

6: Use StandardScaler to apply feature scaling on the training and test sets.

7: Print scaled training and test sets to verify feature scaling.

"""


#### Importing libraries
##"""

import numpy as np
import pandas as pd

#@"""



#### Importing dataset
##"""

dataset = pd.read_csv('datasets/iris.csv')
input_cols = dataset.iloc[ : , : -1].values   # // X
output_col = dataset.iloc[ : , -1].values    # // y

# print(dataset.describe() )
# print(dataset.info() )

#@"""



#### Taking Care of Missing Data
##"""

# print(dataset.isnull().sum() ) # // no null values in this data


#@"""



#### Splitting the Data
##"""

from sklearn.model_selection import train_test_split

# // X_train, X_test, y_train, y_test, (tuple unpacking)
(input_cols_train, input_cols_test,
 output_col_train, output_col_test) = train_test_split(input_cols, output_col, test_size=0.2, random_state=42)

# print("matrix_of_features_train = \n", input_cols_train)
# print("matrix_of_features_test = \n", input_cols_test)
# print("dependent_variable_train = \n", output_col_train)
# print("dependent_variable_test = \n", output_col_test)


#@"""



#### Feature Scaling
##"""

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

input_cols_train[ : , : ] = standard_scaler.fit_transform(input_cols_train[ : , : ] )
input_cols_test[ : , : ] = standard_scaler.transform(input_cols_test[ : , : ] )

print("input_cols_train = \n", input_cols_train)
print("input_cols_test = \n", input_cols_test)

#@"""
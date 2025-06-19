"""
Coding exercise 5: Feature scaling for Machine Learning

Import the necessary libraries for data preprocessing, including the StandardScaler and train_test_split classes.

Load the "Wine Quality Red" dataset into a pandas DataFrame. You can use the pd.read_csv function for this.
Make sure you set the correct delimiter for the file.

Split your dataset into an 80-20 training-test set. Set random_state to 42 to ensure reproducible results.

Create an instance of the StandardScaler class.

Fit the StandardScaler on features from the training set, excluding the target variable 'Quality'.

Use the "fit_transform" method of the StandardScaler object on the training dataset.

Apply the "transform" method of the StandardScaler object on the test dataset.

Print your scaled training and test datasets to verify the feature scaling process.

"""

#### Importing
##"""

import pandas as pd
import numpy as np

#@"""



#### Importing Dataset
##"""

# dataset = pd.read_csv('datasets/winequality-red.csv')
# print(dataset.shape)  # // (1599, 1)
# // the cols are separated by (;) instead of (,) to solve this:
dataset = pd.read_csv('datasets/winequality-red.csv', sep=';')
print(dataset.shape)

# dataset = pd.read_csv('datasets/winequality-red2.csv')
# print(dataset.shape)  # // (1599, 12)

# dataset.drop(index=1598, axis=0, inplace=True)  # // to drop a row
# dataset.drop(["Team", "Weight"], axis = 1, inplace = True) # // to drop cols

X = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , -1].values

# print(dataset.info() )

#@"""



#### Splitting the dataset
##"""

from sklearn.model_selection import train_test_split

(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)

#@"""



#### Feature Scaling
##"""

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
X_train[ : , : ] = standard_scaler.fit_transform(X_train[ : , : ] )
X_test[ : , : ] = standard_scaler.transform(X_test[ : , : ] )

print("X_train = \n", X_train)
print("X_test = \n", X_test)

#@"""

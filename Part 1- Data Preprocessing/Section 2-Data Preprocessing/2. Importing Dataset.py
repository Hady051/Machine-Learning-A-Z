#### Importing libraries
##"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#@"""



#### Importing dataset
##"""

dataset = pd.read_csv('datasets\\Data.csv')
# X = dataset.iloc[ : , :-1].values
# y = dataset.iloc[ : , -1].values
# X = dataset.iloc[ : , :-1]
# y = dataset.iloc[ : , -1]

features = dataset.iloc[ : , :-1].values
dependent_variable = dataset.iloc[ : , -1].values

print("X = ", features)
print("y = ", dependent_variable)

#@"""
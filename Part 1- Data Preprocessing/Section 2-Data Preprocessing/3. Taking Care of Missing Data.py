#### Importing libraries
##"""

import numpy as np
import pandas as pd

#@"""



#### Importing dataset
##"""

dataset = pd.read_csv('datasets\\Data.csv')
features = dataset.iloc[ : , :-1].values    # // X
dependant_variable = dataset.iloc[ : , -1].values # // y

# print(features)
# print(dependant_variable)

#@"""



#### Taking care of Missing data
##"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(features[ : , 1:3])
features[ : , 1:3] = imputer.transform(features[ : , 1:3])

print("The new features matrix = ", features)

#@"""
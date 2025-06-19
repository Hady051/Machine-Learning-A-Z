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

dataset = pd.read_csv('./Datasets/Position_Salaries.csv')
# print(dataset)
'''
            Position  Level   Salary
0   Business Analyst      1    45000
1  Junior Consultant      2    50000
2  Senior Consultant      3    60000
3            Manager      4    80000
4    Country Manager      5   110000
5     Region Manager      6   150000
6            Partner      7   200000
7     Senior Partner      8   300000
8            C-level      9   500000
9                CEO     10  1000000

'''

input_feature = dataset.iloc[ : , 1:-1].values
dv = dataset.iloc[ : , -1].values # // dependent variable

# print(input_feature)
'''
[[ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]]
'''

# print(dv)
'''
[  45000   50000   60000   80000  110000  150000  200000  300000  500000
 1000000]
'''

### Description:
# // ** Same dataset of section 8

#@"""



#### preprocessing
##"""

dv = dv.reshape(len(dv), 1)
# print(dv)
'''
[[  45000]
 [  50000]
 [  60000]
 [  80000]
 [ 110000]
 [ 150000]
 [ 200000]
 [ 300000]
 [ 500000]
 [1000000]]
'''


### Description:
## Line 66: reshaping the DV
'''
# // We would need to transform the dv to a 2D array, where you have the salaries displayed vertically
# / like the input_feature variable
# // Why transform it into 2D?
# // ** Because the scaler whether it's normalization or standardization, expects one unique format in it's  
# / input, which is a 2D array.

# // 1st arg: no. of rows
# // 2nd arg: no. of cols. 
'''

#@"""



#### Feature Scaling (Normalization)
"""

from sklearn.preprocessing import MinMaxScaler

## Normalizing the input feature
normalization_scaler_X = MinMaxScaler()
input_feature = normalization_scaler_X.fit_transform(input_feature)
# print(input_feature)
'''
[[0.        ]
 [0.11111111]
 [0.22222222]
 [0.33333333]
 [0.44444444]
 [0.55555556]
 [0.66666667]
 [0.77777778]
 [0.88888889]
 [1.        ]]
'''

## Normalizing the Dependent Variable
normalization_scaler_y = MinMaxScaler()
dv = normalization_scaler_y.fit_transform(dv)
# print(dv)
'''
[[0.        ]
 [0.0052356 ]
 [0.01570681]
 [0.03664921]
 [0.06806283]
 [0.10994764]
 [0.16230366]
 [0.26701571]
 [0.47643979]
 [1.        ]]

'''

### Description:
# // ** I will try Normalization here, instead of Standardization like in the course
# // we created two scalers, one for each. I described why in the course's NB.
# // In general, we didn't need to create 2 scalers.

"""



#### Feature Scaling (Standardization)
##"""

from sklearn.preprocessing import StandardScaler

## Standardizing the input feature
standard_scaler_X = StandardScaler()
input_feature = standard_scaler_X.fit_transform(input_feature)
# print(input_feature)
'''
[[-1.5666989 ]
 [-1.21854359]
 [-0.87038828]
 [-0.52223297]
 [-0.17407766]
 [ 0.17407766]
 [ 0.52223297]
 [ 0.87038828]
 [ 1.21854359]
 [ 1.5666989 ]]
'''

## Standardizing the Dependent Variable
standard_scaler_y = StandardScaler()
dv = standard_scaler_y.fit_transform(dv)
# print(dv)
'''
[[-0.72004253]
 [-0.70243757]
 [-0.66722767]
 [-0.59680786]
 [-0.49117815]
 [-0.35033854]
 [-0.17428902]
 [ 0.17781001]
 [ 0.88200808]
 [ 2.64250325]]
'''


### Description:
'''
# // we created two scalers, one for each. I described why in the course's NB.
# // In general, we didn't need to create 2 scalers.
'''

#@"""



#### Training the SVR Model
##"""

from sklearn.svm import SVR

supp_vector_regressor = SVR(kernel='rbf')
supp_vector_regressor.fit(input_feature, np.ravel(dv) )

### Description:
## Line 204: creating an SVR object
'''
# // we specified the RBF kernel by naming it inside as the "kernel" input.
# // There are other SVM/SVR kernels as well.
'''

## Line 205: fitting the SVR model
'''
DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)

# // I guess we should reshape the dv to be 1D, use np.ravel() or .ravel() or flatten() methods
'''

#@"""



#### Predicting a new result
##"""

scaled_prediction = supp_vector_regressor.predict(standard_scaler_X.transform([[6.5]]) )
# print(scaled_prediction)
'''
[-0.27861589]
'''

scaled_prediction = scaled_prediction.reshape(-1, 1)
# print(scaled_prediction)
'''
[[-0.27861589]]
'''

print("SVR model prediction is: ",
      standard_scaler_y.inverse_transform(scaled_prediction) )

## O
'''
SVR model prediction is:  [[170370.0204065]]
'''

## after Normalization FS:
# -> SVR model prediction is:  [[241230.33748789]]
# // different value than that of the video, because, I used Normalization instead of Standardization.

## after Standardization FS:
# -> SVR model prediction is:  [[170370.0204065]]

### Description:
## Line of the Video not the one here
'''
# // we will start with the scaler "standard_scaler_X" to transform the input to the same scaling of the "levels" col.

# // ** reshape():
-1: This means that the number of rows is inferred from the length of the array and the remaining dimensions.
    Essentially, it tells NumPy to calculate the appropriate number of rows automatically.
    ** I think "-1" gives the same effect of len().
1: This specifies that the array should have one column.

the "standard_scaler_y" to get the output of the prediction scaled according to
# / the "salaries" col. scale.

# // we will use "inverse_transform()" method to get the output as the actual salary and not a scaled salary.
# // basically transforming the scaled salary back to the actual salary.

'''

#@"""



#### Visualizing the SVR model results
##"""

## real points
plt.scatter(standard_scaler_X.inverse_transform(input_feature),
            standard_scaler_y.inverse_transform(dv),
            color="orange")
## LR prediction (plotting the line of the LR predictions)
plt.plot(standard_scaler_X.inverse_transform(input_feature),
         standard_scaler_y.inverse_transform(supp_vector_regressor.predict(input_feature).reshape(-1, 1) ),
         color="blue")

plt.title("SVR predictions with the real values [Standardization]")
plt.xlabel("Position Label.")
plt.ylabel("Salary.")

plt.show()

#@"""



#### Notes:
##"""

# // **I tried "Normalization" at first as a feature scaling method.
# // I had a very high prediction, and in general, the predictions weren't good.

# // ** I then tried "Standardization".

#@"""















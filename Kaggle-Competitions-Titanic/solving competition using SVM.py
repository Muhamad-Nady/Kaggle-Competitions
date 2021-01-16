# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

#predicting age of train data
from sklearn.preprocessing import Imputer
age =  train.iloc[:,5] 
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
age =  imp.fit_transform(age)
age = age.transpose()
train.iloc[:,5] = age

# predicting age of test data          
test_age =  test.iloc[:,4] 
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
test_age =  imp.fit_transform(test_age)
test_age = test_age.transpose()
test.iloc[:,4] = test_age
         
# encoding sex in train data
X_train = train.iloc[:, [ 2, 4,5, 6, 7, 9]]
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_train.iloc[:,1] = labelencoder_X.fit_transform(X_train.iloc[:,1])                              
y_train = train.iloc[:, 1].values 
             
# encoding sex in test data              
X_test = test.iloc[:, [ 1, 3, 4, 5, 6, 8]]
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_test.iloc[:,1] = labelencoder_X.fit_transform(X_test.iloc[:,1])

#predacting x_test['fare'] values
from sklearn.preprocessing import Imputer
fare =  X_test.iloc[:,5] 
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
fare =  imp.fit_transform(fare)
fare = fare.transpose()
X_test.iloc[:,5] = fare                 
                  
"""X_test.iloc[152,5] = np.sum(X_test.iloc[:,5])/418
X_test = X_test.astype(np.float64)"""

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(gender_submission.iloc[:,1], y_pred)

# transform predicted data to gender_submission file
gender_submission.iloc[:,1] = y_pred 
                      
# saving csv file
from pandas import DataFrame
gender_submission.to_csv('gender_submission_pred.csv',encoding='utf-8', index = False)
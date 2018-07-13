#importing the libraries
import numpy as np  #CONTAINS MATHS TOOLS
import matplotlib.pyplot as plt #PLOTTING CHARTS
import pandas as pd #IMPORTING AND MANAGING DATA

#importing the dataset
dataset= pd.read_csv('train.csv', index_col=0)
dataset = dataset.drop(['Name', 'Cabin', 'Ticket'], axis=1)

X=dataset.iloc[:,1:].values #INDEPENDENT VARIABLES
Y=dataset.iloc[:,0].values   #DEPENDENT VARIABLES

#Handling missing data
from sklearn.preprocessing import Imputer #IMPUTER HANDLES MISSING DATA
imputer= Imputer(missing_values= 'NaN', strategy= 'mean' ,axis =1)
#TODO IMPUTER=IMPUTER.. WHY?
imputer=imputer.fit(X[:,2]) #upper bound is not included, this means col 2 
X[:,2]=imputer.transform(X[:,2])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
X[:,1]=number.fit_transform(X[:,1].astype('str'))
X[:,6]=number.fit_transform(X[:,6].astype('str'))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder() #LABELENCODER HELPS TO CATEGORISE DATA INTO 0,1,2 ..
X[:,0] =labelencoder_X.fit_transform(X[:,0])
X[:,6] =labelencoder_X.fit_transform(X[:,6]) 
onehotencoder=OneHotEncoder(categorical_features = [0]) #ONEHOTENCODER TRANFORMS THE CATEGORIACL DATA INTO VECTORS OF O AMD 1S
X = onehotencoder.fit_transform(X).toarray()

#Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y , test_size= 0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty='l1')
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(classifier.score(X_train, Y_train))
print(classifier.score(X_test, Y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn import model_selection
import pickle
filename = 'finalized_model_logistic_regression.sav'
pickle.dump(classifier, open(filename, 'wb'))
 

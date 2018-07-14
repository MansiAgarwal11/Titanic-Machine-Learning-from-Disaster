#importing the libraries
import numpy as np  #CONTAINS MATHS TOOLS
import matplotlib.pyplot as plt #PLOTTING CHARTS
import pandas as pd #IMPORTING AND MANAGING DATA
from sklearn.cluster import KMeans

#importing the dataset
dataset= pd.read_csv('train.csv', index_col=0)
#getting rid of unnecessary attributes
dataset = dataset.drop(['Name', 'Cabin', 'Ticket'], axis=1)

#print(dataset.isnull().any())  
#dropping rows with nan values in embarked attribute
dataset = dataset.dropna(subset=['Embarked']) 

#partitioning data into dependent and independent
X=dataset.iloc[:,1:].values 
Y=dataset.iloc[:,0].values

#Handling missing data for the true attributes
from sklearn.preprocessing import Imputer 
imputer= Imputer(missing_values= 'NaN', strategy= 'mean' ,axis =1)
imputer=imputer.fit(X[:,2]) #age
X[:,2]=imputer.transform(X[:,2])

#Encoding categorical data-string datatype
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
X[:,1]=number.fit_transform(X[:,1].astype('str'))
X[:,6]=number.fit_transform(X[:,6].astype('str'))

#Encoding categorical data-int datatype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder() 
X[:,0] =labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features = [0,6]) 
X = onehotencoder.fit_transform(X).toarray()

#Splitting the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y , test_size= 0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,7:] = sc.fit_transform(X_train[:,7:])
X_test[:,7:] = sc.fit_transform(X_test[:,7:])

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(classifier.score(X_train, Y_train))
print(classifier.score(X_test, Y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn import model_selection
import pickle
filename = 'finalized_model_naive.sav'
pickle.dump(classifier, open(filename, 'wb'))

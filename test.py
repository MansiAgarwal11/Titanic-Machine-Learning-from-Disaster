#importing the libraries
import numpy as np  #CONTAINS MATHS TOOLS
import matplotlib.pyplot as plt #PLOTTING CHARTS
import pandas as pd #IMPORTING AND MANAGING DATA

#importing the dataset
dataset= pd.read_csv('test.csv', index_col=0)
dataset = dataset.drop(['Name', 'Cabin', 'Ticket'], axis=1)

X=dataset.iloc[:,:].values #INDEPENDENT VARIABLES

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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = sc.transform(X)

from sklearn import model_selection
import pickle
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X, Y)
print(result)
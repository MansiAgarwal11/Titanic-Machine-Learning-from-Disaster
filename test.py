#importing the libraries
import numpy as np  #CONTAINS MATHS TOOLS
import matplotlib.pyplot as plt #PLOTTING CHARTS
import pandas as pd #IMPORTING AND MANAGING DATA

#importing the dataset
dataset= pd.read_csv('test.csv', index_col=0)
#dropping useless attributes
dataset = dataset.drop(['Name', 'Cabin', 'Ticket'], axis=1)

#print(dataset.isnull().any()) 

X=dataset.iloc[:,:].values #INDEPENDENT VARIABLES
 
#Handling missing data
from sklearn.preprocessing import Imputer 
imputer= Imputer(missing_values= 'NaN', strategy= 'mean' ,axis =1)
imputer=imputer.fit(X[:,2]) #age
X[:,2]=imputer.transform(X[:,2])
imputer=imputer.fit(X[:,5]) #fare
X[:,5]=imputer.transform(X[:,5])

#Encoding categorical data-string datatype
from sklearn.preprocessing import LabelEncoder
number= LabelEncoder()
X[:,1]=number.fit_transform(X[:,1].astype('str')) #sex
X[:,6]=number.fit_transform(X[:,6].astype('str')) #embarked

#Encoding categorical data -int datatype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder() 
X[:,0] =labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features = [0,6]) 
X = onehotencoder.fit_transform(X).toarray()

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:,7:] = sc.fit_transform(X[:,7:])

#predicting using model
from sklearn import model_selection
import pickle
filename = 'finalized_model_randforest.sav'
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X)

#creating csv file
d= pd.read_csv('test.csv')
d = d.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Fare', 'Cabin', 'Ticket', 'Embarked'], axis=1)
d['Survived'] = y_pred
csv_name='RandForest'
d.to_csv(csv_name, index=False)

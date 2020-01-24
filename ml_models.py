# Imports for ML Models
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ML Models

# Reading the training data
biopslides=pd.read_csv('Train.csv')
X = biopslides.drop('y', axis=1)  
y = biopslides['y']

#Splitting the data into Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.60)

#Declaring the SVM Classifier
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)


#Declaring the Logistic Regression Classifier
logreg = LogisticRegression(C=100).fit(X_train, y_train)

svm_objects = (svclassifier, X_train, X_test, y_train, y_test)
lr_objects = (logreg, X_train, X_test, y_train, y_test)

# Saving the SVM Model
svm_model = 'svm_model.sav'
pickle.dump(svm_objects, open(svm_model, 'wb'))

# Saving the LR Model 
lr_model = 'lr_model.sav'
pickle.dump(lr_objects, open(lr_model, 'wb'))






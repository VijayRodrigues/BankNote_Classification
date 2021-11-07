import pandas as pd
import numpy as np
import pickle
from pyforest import*

df = pd.read_csv('BankNote_Authentication.csv')


X = df.drop(['class'], axis = 1)
y = df['class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rdf_classifier=RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)
rdf_classifier.fit(X_train, y_train)

pickle.dump(rdf_classifier , open('BankNote.pkl', 'wb'))
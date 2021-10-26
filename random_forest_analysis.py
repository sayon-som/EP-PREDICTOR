# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:49:57 2021

@author: user
"""
#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#IMORTING THE DATA SET
dataset = pd.read_csv('CCPP.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
#SPLITTING THE DATA SET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
#TRAINING THE MODEL
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
reg.fit(X_train, y_train)
#PREDICTION
y_pred = reg.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#rsquare evaluation of the model
from sklearn.metrics import r2_score
r2_value=r2_score(y_test, y_pred)
print(r2_value)
#as the r2 val is close to 1 , so the data is well adapted  to the random forest regression model.
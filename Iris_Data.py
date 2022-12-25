# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 13:54:31 2022

@author: joses
"""
from sklearn.datasets import load_iris
#%%

X,y = load_iris(return_X_y=True)

#%%

import sklearn.svm as svm

#%%

SVC = svm.LinearSVC()

#%%

from sklearn.model_selection import train_test_split

#%%

X_train,X_test,y_train,Y_hat = train_test_split(X,y,random_state=42)
model = SVC.fit(X_train,y_train)

#%%

y_pred = model.predict(X_test)

#%%

import sklearn.metrics as metrics

#%%

print(metrics.accuracy_score(Y_hat, y_pred))

#%%
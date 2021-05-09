# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 21:00:06 2021

@author: ABRA
"""
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veriyi yükleme
veriler = pd.read_csv('Churn_Modelling.csv')

#veri on isleme

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Artificial Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
    #layers
classifier.add(Dense(units=6,kernel_initializer='uniform', activation = "relu",input_dim=11))
classifier.add(Dense(units=6,kernel_initializer='uniform', activation = "relu"))
classifier.add(Dense(units=1,kernel_initializer='uniform', activation = "sigmoid"))

    #Optimizasyon
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=50)
y_pred = classifier.predict(X_test)


y_pred_binary = (y_pred > 0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_binary)

print(cm)




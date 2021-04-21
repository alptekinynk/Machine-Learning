#kütüphane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri ön-işleme
veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken

#verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) #sc'ye zaten x_train için fit uygulandı


#lojistik regression
from sklearn.linear_model import LogisticRegression

log_r = LogisticRegression(random_state=0)
log_r.fit(X_train,y_train)

#prediction
y_pred = log_r.predict(X_test)
print(y_pred)
print(y_test)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)
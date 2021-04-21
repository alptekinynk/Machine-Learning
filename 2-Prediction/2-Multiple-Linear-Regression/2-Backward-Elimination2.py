#Dataframe oluşturulması ve bölünmesi

#import kısmı
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

#verilerin alınması
veriler = pd.read_csv("tenis.csv")

diger = veriler.iloc[:,1:3].values
#Kategorik cinsiyet ve ülke verisi sayısal veriye çevirildi
from sklearn.preprocessing import LabelEncoder
outlook = veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(veriler.iloc[:,0:1])
OHE = preprocessing.OneHotEncoder()
outlook = OHE.fit_transform(outlook).toarray()

windy = veriler.iloc[:,3:4].values
le2 = preprocessing.LabelEncoder()
windy[:,0] = le2.fit_transform(veriler.iloc[:,3:4])
OHE2 = preprocessing.OneHotEncoder()
windy = OHE2.fit_transform(windy).toarray()

play = veriler.iloc[:,-1:].values
le3 = preprocessing.LabelEncoder()
play[:,-1] = le3.fit_transform(veriler.iloc[:,-1])
OHE3 = preprocessing.OneHotEncoder()
play = OHE2.fit_transform(play).toarray()

#DataFrame oluşturulması
sonuc1 = pd.DataFrame(data=outlook,index=range(14),columns=["rainy","overcast","sunny"])

sonuc2 = pd.DataFrame(data=diger[:,:-1], index=range(14),columns=["temperature"])

sonuc3 = pd.DataFrame(data=windy[:,-1:], index=range(14),columns=["windy"])

sonuc4 = pd.DataFrame(data=play[:,-1:], index=range(14),columns=["play"])

sonuc5 = pd.DataFrame(data=diger[:,-1:], index=range(14),columns=["humidity"])


#DataFrame birleştirilmesi
genel = pd.concat([sonuc1,sonuc2],axis=1)

genel = pd.concat([genel,sonuc3],axis=1)

genel = pd.concat([genel,sonuc4],axis=1)

genel1 = pd.concat([genel,sonuc5],axis=1)

#Test ve eğitim verisinin bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(genel,sonuc5,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

#BACKWARD ELİMİNATİON
import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = genel, axis=1)

x_list = genel.iloc[:,[0,1,2,3,5]].values

x_list = np.array(x_list,dtype=float)

model = sm.OLS(sonuc5,x_list).fit()

print(model.summary()) 


genel = genel.iloc[:,[0,1,2,3,5]].values

x_train, x_test, y_train, y_test = train_test_split(genel,sonuc5,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

lr2 = LinearRegression()
lr2.fit(x_train,y_train)

y_pred2 = lr2.predict(x_test)

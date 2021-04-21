#import kısmı
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
#verileri aldık
veriler = pd.read_csv("maaslar_yeni.csv")

#Data Frame Slicing

maas = veriler.iloc[:,5:]#Maas
genel = veriler.iloc[:,2:5]#ünvan_seviyesi,kıdem,puan

MAAS = maas.values
GENEL = genel.values

#Linear Regression
from sklearn.linear_model import LinearRegression

LR= LinearRegression()
LR.fit(GENEL,MAAS)

    #backward elimination
model=sm.OLS(LR.predict(GENEL),GENEL)
print(model.fit().summary())
print("Linear R2 değeri")
print(r2_score(MAAS,LR.predict(GENEL)))

print("---------------------------------")


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
PR = PolynomialFeatures(degree = 5)
genel_p = PR.fit_transform(GENEL)
print(genel_p)

LR2 = LinearRegression()
LR2.fit(genel_p,maas)

 #poly tahmin

print('poly OLS')
model2=sm.OLS(LR2.predict(PR.fit_transform(GENEL)),GENEL)
print(model2.fit().summary())

print('Polynomial R2 degeri')
print(r2_score(MAAS,LR2.predict(PR.fit_transform(GENEL))))


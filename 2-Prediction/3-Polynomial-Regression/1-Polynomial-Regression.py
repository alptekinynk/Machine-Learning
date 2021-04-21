#Dataframe oluşturulması ve bölünmesi

#import kısmı
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#verilerin alınması
veriler = pd.read_csv("maaslar.csv")


#Data Frame slicing
x = veriler.iloc[:,1:2] #eğitim seviyesi
y = veriler.iloc[:,2:] #maaşlar

    #NumPY array 
X = x.values
Y = y.values

#Linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)



#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
'''
Linear regression modeli ile polinom oluşturmadan önce veriyi polinomal çeviri
ile çeviriyoruz ve daha sonra linear regression'a veriyi gönderiyoruz
'''
pf = PolynomialFeatures(degree = 2)
x_poly = pf.fit_transform(X)
lr_2 = LinearRegression()
lr_2.fit(x_poly,y)



    #polynomial regression modelimizin derecesini 4 olarak ayarladık
pf3 = PolynomialFeatures(degree = 4)
x_poly3 = pf3.fit_transform(X)
lr_3 = LinearRegression()
lr_3.fit(x_poly3,y)




#Plotting
plt.scatter(X,Y, color='red')
plt.plot(x,lr.predict(X), color='blue')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lr_2.predict(pf.fit_transform(X)),color = 'blue')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(X,lr_3.predict(pf3.fit_transform(X)),color = 'blue')
plt.show()

#Predict
'''
eğitim seviyesi 11 olan ve 8.9 olan birine ne kadar maaş verileceğini Linear regression ıle tahmin ettik
'''
print(lr.predict([[11]]))
print(lr.predict([[8.9]]))

'''
aynı tahmini polinomal regression ıle gerçekleştirdik
'''
print(lr_2.predict(pf.fit_transform([[11]])))
print(lr_2.predict(pf.fit_transform([[8.9]])))
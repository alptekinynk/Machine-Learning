#import kısmı
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

#VERİLERİN ALINMASI
veriler = pd.read_csv("satislar.csv")

#VERİ ÖN İŞLEME

   
aylar = veriler[['Aylar']]        
satislar = veriler[['Satislar']]

 #Veriler bu şekilde de alınabilir.
#aylar = veriler.iloc[:,0:0].values
#satislar = veriler.iloc[:,1:2].values



#VERİLERİN BÖLÜNMESİ

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)


'''
#Verilerin Ölçeklendirilmesi

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
    
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''

#Model inşaası (Linear Regression)

from sklearn.linear_model import LinearRegression

lr = LinearRegression() #objeyi oluşturduk.

lr.fit(x_train,y_train) #makinemizin x_train den y_train'i öğrenmesini istedik.

tahminler = lr.predict(x_test) #öğrendiklerini x_test'i kullanarak y_test'i tahmin etmesini istedik.

    #veri grafiğinin düzgün çıkması için verileri sıralıyoruz
x_train = x_train.sort_index() 
y_train = y_train.sort_index()

#Tahmin tablosunu ve gerçek değerleri grafiğe çeviriyoruz
plt.title("Aylara Göre Satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.plot(x_train,y_train)
plt.plot(x_test,tahminler)





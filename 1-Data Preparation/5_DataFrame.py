#Dataframe oluşturulması ve birleştirilmesi

#import kısmı
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

#verilerin alınması
veriler = pd.read_csv("veriler.csv")

#eksik verilerin tespiti ve doldurulması
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

Yas = veriler.iloc[:,1:4].values

imputer = imputer.fit_transform(Yas[:,1:4])

#Kategorik verinin dönüştürülmesi
ulke = veriler.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1])

OHE = preprocessing.OneHotEncoder()

ulke = OHE.fit_transform(ulke).toarray()

#DataFrame oluşturulması
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["FR","TR","US"])

sonuc2 = pd.DataFrame(data=Yas, index=range(22),columns=["BOY","KİLO","YAS"])

sonuc3 = pd.DataFrame(data=veriler.iloc[:,-1].values,index=range(22),columns=["CİNSİYET"])

#DataFrame birleştirilmesi
genel1=pd.concat([sonuc,sonuc2],axis=1)

genel=pd.concat([genel1,sonuc3],axis=1)

print(genel)
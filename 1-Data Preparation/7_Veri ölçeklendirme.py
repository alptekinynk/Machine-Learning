#Dataframe oluşturulması ve bölünmesi

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

    #verileri ayırmak için gerekli kütüphaneyi ekliyoruz

from sklearn.model_selection import train_test_split
    #verilerin %33'lük bir kısmını test için kullarnırken geri kalan kismini train için kullanıyoruz.
    #burada DataFrame'i 4 e böldük.Farklı sütunlar için farklı test ve train değeleri oluşturuyoruz.
x_train, x_test, y_train, y_test = train_test_split(genel1,sonuc3,test_size=0.33, random_state=0)

#Verilerin Ölçeklendirilmesi
    #Sütunlarda ki veriler farklı aralıklarda olabilir dolayısıyla bu verileri aynı aralığa ve orana getirmek için ölçeklendirmemiz lazım
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
    
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)



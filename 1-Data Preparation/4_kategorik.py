#Kategorik verilerin sayısal veriye çevirilmesi

#part import
import pandas as pd
from sklearn import preprocessing

#part data
veriler = pd.read_csv("veriler.csv")

#part kod
"veriler içerisinde ulke sütununu almak için iloc kullandık"
ulke = veriler.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

"Label encoder ile kategorik halde bulunan veriyi sayısal hale çevirdik"
ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1])
print(ulke)

"OHE ile sayısal halde bulunan tek sıralı array'i matris haline çeviriyoruz"
ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)
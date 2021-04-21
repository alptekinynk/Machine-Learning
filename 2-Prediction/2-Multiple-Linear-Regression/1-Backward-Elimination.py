#Dataframe oluşturulması ve bölünmesi

#import kısmı
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

#verilerin alınması
veriler = pd.read_csv("veriler.csv")

#veri ön işleme

Yas = veriler.iloc[:,1:4].values

#Kategorik cinsiyet ve ülke verisi sayısal veriye çevirildi
from sklearn.preprocessing import LabelEncoder
ulke = veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0:1])

OHE = preprocessing.OneHotEncoder()
ulke = OHE.fit_transform(ulke).toarray()


c = veriler.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

OHE = preprocessing.OneHotEncoder()
c = OHE.fit_transform(c).toarray()


#DataFrame oluşturulması
sonuc = pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])

sonuc2 = pd.DataFrame(data=Yas, index=range(22),columns=["boy","kilo","yas"])

cinsiyet = veriler.iloc[:,-1].values
    #Dummy variable olmaması için datayı ayırdık
sonuc3 = pd.DataFrame(data=c[:,:1],index=range(22),columns=["cinsiyet"])

#DataFrame birleştirilmesi
genel = pd.concat([sonuc,sonuc2],axis=1)

genel1 = pd.concat([genel,sonuc3],axis=1)



#Test ve eğitim verisinin bölünmesi

from sklearn.model_selection import train_test_split


    #Genel1 tablosundan cinsiyet verisini ayırdık ve makinenin öğrenmesi için y_train olarak ayarladık.
x_train, x_test, y_train, y_test = train_test_split(genel,sonuc3,test_size=0.33, random_state=0)


    #Tahmini gerçekleştirmek için Linear regression onjesini oluşturduk.
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

    #Boy tahmini için boy verisini ayırdık 
boy = genel1.iloc[:,3:4].values

sol = genel1.iloc[:,:3] #yeni bir dataFrame oluşturmak için genel tablodan boy verisinin sol va sağını aldık
sag = genel1.iloc[:,4:5]

veri = pd.concat([sol,sag],axis=1)
    #Cinsiyet ülke yaş kilo verilerine bakarak boy tahmini yaptık
x_train, x_test, y_train, y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

lr2 = LinearRegression()
lr2.fit(x_train,y_train)

y_pred2 = lr2.predict(x_test)

#BACKWARD ELİMİNATİON
    #"Veri" tablosu içerisinde hangi parametlerin daha etkili olduğunu öğrenmek için veri başarısını ölçüyoruz.(P.value kullanıyoruz.)
import statsmodels.api as sm
    #error rate'i tablo ya eklemek için aşağıdaki kodu yazıyoruz
'''ones tüm değerleri 1 olan bir array oluşturur.
astype oluşturalan arrayin type'ını belirler. axis ise default olarak 
satır oluşturmaması için 1'e eşitlenir ki bu da onu sütun yapar.'''
X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis=1)

'''ilk başta bütün sutünları x_list'e ekliyoruz.Fakat ilerleyen süreçte tek tek
çıkartarak en optimal değişkenleri bulacağız.'''

x_list = veri.iloc[:,[0,1,2,3]].values
x_list = np.array(x_list,dtype=float)
'''x_list içerisindeki parametrelerin boy üzerine etkisini ölçmek amacı ile model oluşturuyoruz'''
model = sm.OLS(boy,x_list).fit()
print(model.summary()) 


'''
İlk rapor sonucu, x5(yas) değişkeninin p değeri 0.717 çıktı. Dolayısıyla Backward elimination gereği
en yüksek p değeri olan variable'ı tablomuzdan çıkartıyoruz.'''

'''
İkinci rapor sonucu, x5(cinsiyet) değişkeninin p değeri 0.031 çıktı. Bu değer SLdeğeri olan 0.05'in altında olduğu 
için bırakılabilir. Fakat tercihen bu değeri de çıkartıyorum.'''



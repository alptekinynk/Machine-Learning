# -*- coding: utf-8 -*-
#import part
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#verilerin alınması
veriler = pd.read_csv("eksikveriler.csv")


"""
imputer isminde bir obje oluşturuyoruz ve 
consturucter ı çağırıp parametleri veriyoruz.
"""
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')


#iloc = integer location
"""
tüm satırları alması için : koyduk ve sonra 1 den 4 e kadar olan 
kolonları alması için aralık belirttik 1:4 şeklinde
"""
Yas = veriler.iloc[:,1:4].values

#öğrenme aşaması

imputer = imputer.fit(Yas[:,1:4])
"""
fit methodu öğrenmek istediğimiz kısmı daha önce consturctor da 
verdiğimiz strategy e göre çağırıyoruz.
"""

#öğrendiğini uygulama
"""
fit ile öğrendiğimiz değerleri eksik olan verilerin yerine 
doldurulması için transform methodunu çağırıyoruz.
"""

Yas[:,1:4]= imputer.transform(Yas[:,1:4])

print(Yas)
#kütüphaneler
import numpy as np
import pandas as pd
import re
# veri yukleme
yorumlar = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')


#STOPWORDS VE STEMMİNG İŞLEMİ İÇİN GEREKLİ İŞLEMLER
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords

#Preprocessing
yorumlar_2 = []
for i in range(1000):
    yorum = re.sub('[^a-zA-Z]',' ',yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    yorumlar_2.append(yorum) 
    
#Feautre Extraction     
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)

x = cv.fit_transform(yorumlar_2).toarray() #bağımzsız
y = yorumlar.iloc[:,1].values #bağımlı


#Native Bayes
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


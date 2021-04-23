#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin alınması ve ayrılması
veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

#K-Means
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters = 2, init = 'k-means++')
Y_pred = kmeans.fit_predict(X)
print(Y_pred)


print(kmeans.cluster_centers_)

#Optimal K değerini bulma
sonuclar = []
for i in range(1,11):
    kmeans= KMeans(n_clusters = i , init='k-means++', random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,11),sonuclar)
plt.show()

#Görselleştirme
plt.scatter(X[Y_pred==0,0],X[Y_pred==0,1],s=100,c='red')
plt.scatter(X[Y_pred==1,0],X[Y_pred==1,1],s=100,c='green')
#plt.scatter(X[Y_pred==2,0],X[Y_pred==2,1],s=100,c='blue')
plt.title('HC')
plt.show()
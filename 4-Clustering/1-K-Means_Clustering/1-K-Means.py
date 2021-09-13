#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin alınması ve ayrılması
veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

#K-Means
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters = 3, init = 'k-means++')
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
"""
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

X = Monetary
y = Frequency

estimators = [('k_means_8', KMeans(n_clusters=8)),
              ('k_means_3', KMeans(n_clusters=3))]

fignum = 1
titles = ['8 clusters', '3 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(5,5))
    ax = Axes3D(fig, rect=[8, 8, 2, 5], elev=20, azim=120)
    est.fit(X)
    labels = est.labels_

    ax.scatter(Monetary, Frequency, Recency,c=labels,marker = "D", edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Monetary')
    ax.set_ylabel('Frequency ')
    ax.set_zlabel('Recency ')
    ax.set_title(titles[fignum-1 ])
    ax.dist = 10
    fignum = fignum + 1
"""
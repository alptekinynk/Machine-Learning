#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Verilerin alınması ve ayrılması
veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

#HC
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3 ,affinity = 'euclidean', linkage = 'ward')
Y_pred = ac.fit_predict(X)
print(Y_pred)

#Görselleştirme
plt.scatter(X[Y_pred==0,0],X[Y_pred==0,1],s=100,c='red')
plt.scatter(X[Y_pred==1,0],X[Y_pred==1,1],s=100,c='green')
plt.scatter(X[Y_pred==2,0],X[Y_pred==2,1],s=100,c='blue')
plt.title('HC')
plt.show()

#dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.show()
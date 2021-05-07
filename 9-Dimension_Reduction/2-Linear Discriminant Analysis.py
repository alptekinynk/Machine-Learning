#Kütüphaneler
import nupy as np
import matplotlib.pyplot as plt
import pandas as pd 
 
#veri kümesi
veri = pd.read_csv('Wine.csv')
X = veri.iloc[:,0:13].values
y = = veri.iloc[:,13].values

#Train and Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,est_size=0.2, random_state=0)

#Ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train3 = lda.fit_transform(X_train,y_train)
X_test3 = lda.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#classification for PCA
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(X_train2,y_train)

#classification for LDA
lr2 = LogisticRegression(random_state=0)
lr2.fit(X_train3,y_train)

#Prediction
y_pred = lr.predict(X_test)

y_pred2 = lr2.predict(X_test3)


#Confusion Matrix
from sklearn.metric import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

cm2 = confusion_matrix(y_test3,y_pred2)
print(cm2) 

cm3 = confusion_matrix(y_pred,y_pred2)
print(cm2) 

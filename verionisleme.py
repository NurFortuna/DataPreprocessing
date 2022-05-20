# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:11:16 2022

@author: acer
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as lt

#veri onisleme

veriler=pd.read_csv("eksikveriler.csv")

#test
print(veriler)

boy=veriler[['boy']]
print(boy)


boykilo=veriler[['boy','kilo']]
print(boykilo)


#eksik veriler

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy="mean")

Yas=veriler.iloc[:,1:4].values
print(Yas)

imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

#encoder Nominal Ordinal -> Numeric

ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()

print(ulke)


#numpy dizileri dataframe dönüşümü
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)


sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)


cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

#dataframe birleştirme işlemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)


#verilerin egitim ve test için bolunmesi
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

















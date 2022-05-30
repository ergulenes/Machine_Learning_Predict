#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:37:50 2022

@author: enes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 16:57:43 2022

@author: enes
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Crop_recommendation = pd.read_csv('Crop_recommendation.csv')

from sklearn.preprocessing import LabelEncoder
Crop = Crop_recommendation.apply(LabelEncoder().fit_transform) #Verilerin tamamının sayısal hale döndürülmesi
                  
print(Crop)

d = Crop.iloc[:,7:8] #Sayısal hale dönüştürülen kategorik verilerin ayrıştırılması

from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
d2 = hot.fit_transform(d).toarray()

#one hot encoder ile sayısallaştırılan kategorik verilen 0 - 1 şeklinde düzenlenip ayrı sütunlara yerleştirilmesi
plantype = pd.DataFrame(data = d2, index = range(2200), columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s','t','u','y',"z"])


d22 = pd.concat([Crop_recommendation,plantype.iloc[:,:22]],axis = 1)
del d22['label'] #label sütununu sildim

#veriler train ve test olacak şekilde 2'ye bölündü. Azot için tahmin yapılacak(y değeri)
from sklearn.model_selection import train_test_split
x_tr, x_tt,y_tr,y_tt = train_test_split(d22.iloc[:,1:],d22.iloc[:,:1],test_size=0.33, random_state=0)

#bazı değerlerin silinmesi
del x_tr['P']
del x_tt['P']
del x_tr['K']
del x_tt['K']

#verilerin values dönüşümünün yapılması. Yoksa "random.fit(x_tr,y_tr.ravel())" burada hata veriyor.
x_tr = x_tr.values
x_tt = x_tt.values
y_tr = y_tr.values
y_tt = y_tt.values

#verilerin eğitilmesi
from sklearn.ensemble import RandomForestRegressor
random = RandomForestRegressor(n_estimators=999, random_state=0)
random.fit(x_tr,y_tr.ravel())

#x_tt'ye göre y_tt'nin tahmini
y_est = random.predict(x_tt)
print(y_est)



#R^2 değerinin hesaplanması
from sklearn.metrics import r2_score
print("Value of R2 for Linear Regression")
print(r2_score(y_tt,y_est))

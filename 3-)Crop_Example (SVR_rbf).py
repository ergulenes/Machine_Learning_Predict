#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:19:19 2022

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

plantype = pd.DataFrame(data = d2, index = range(2200), columns=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s','t','u','y',"z"])
#one hot encoder ile sayısallaştırılan kategorik verilen 0 - 1 şeklinde düzenlenip ayrı sütunlara yerleştirilmesi

d22 = pd.concat([Crop_recommendation,plantype.iloc[:,:22]],axis = 1)
del d22['label']
#label sütununu sildim

from sklearn.model_selection import train_test_split
x_tr, x_tt,y_tr,y_tt = train_test_split(d22.iloc[:,1:],d22.iloc[:,:1],test_size=0.33, random_state=0)
#veriler train ve test olacak şekilde 2'ye bölündü. Azot için tahmin yapılacak(y değeri)
del x_tr['P']
del x_tt['P']
del x_tr['K']
del x_tt['K']

#verilerin scaler ile 0 - 1 arasında ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

scale1=StandardScaler() 
x_tr = scale1.fit_transform(x_tr)

scale2=StandardScaler()
x_tt = scale2.fit_transform(x_tt)

y_tt = scale2.fit_transform(y_tt)

y_tr = scale2.fit_transform(y_tr)

"""
scale3=StandardScaler()
x_tr = scale1.fit_transform(y_tr)

sclae4=StandardScaler()
x_tt = scale1.fit_transform(x_tt)
"""

#verilerin eğitilmesi
from sklearn.svm import SVR

reg_svr = SVR(kernel="rbf")
reg_svr.fit(x_tr,y_tr)



#x_tt'ye göre y_tt'nin tahmini
y_est = reg_svr.predict(x_tt)

"""
#verilerin sıralanması
x_tr = x_tr.sort_index()
y_tr = y_tr.sort_index()
"""



#R^2 değerinin hesaplanması
from sklearn.metrics import r2_score
print("Value of R2 for Linear Regression")
print(r2_score(y_tt,y_est))


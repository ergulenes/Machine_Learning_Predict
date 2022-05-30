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
""""
del x_tr['P']
del x_tt['P']
del x_tr['K']
del x_tt['K']
"""

#verilerin eğitilmesi
from sklearn.preprocessing import PolynomialFeatures
pol_reg=PolynomialFeatures(degree=1)
x_tr_pol=pol_reg.fit_transform(x_tr) #burada x eğitim verisi polinomal denklem haline getirilmiştir.
x_tt_pol = pol_reg.fit_transform(x_tt) #burada x test verisi polinomal denklem haline getirilmiştir.
print(x_tr_pol)

#linear regrasyon dönüşümünün uygulanması
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_tr_pol,y_tr) #polinomal x_tr verisi linear regresyon ile eğitim yapılmıştır
y_est = linreg.predict(x_tt_pol) #polinomal x_tt verisi kullanılarak y_tt verisi tahmin edilmiştir.

print(y_est)

#R^2 değerinin hesaplanması
from sklearn.metrics import r2_score
print("Value of R2 for Linear Regression")
print(r2_score(y_tt,y_est))

"""
x_pol = x_pol.fit(x_tr,y_tr)

y_predict = x_pol.predict()


#x_tt'ye göre y_tt'nin tahmini
"""

"""
#verilerin sıralanması
x_tr = x_tr.sort_index()
y_tr = y_tr.sort_index()
"""


"""
#R^2 değerinin hesaplanması
from sklearn.metrics import r2_score
print("Value of R2 for Linear Regression")
print(r2_score(y_tt,y_est))
"""
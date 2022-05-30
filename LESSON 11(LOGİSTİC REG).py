#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:49:32 2022

@author: enes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv("datas.csv")

a = datas.iloc[:,1:4].values
b = datas.iloc[:,4:].values
print(b)


from sklearn.model_selection import train_test_split

a_tr, a_tt, b_tr, b_tt = train_test_split(a,b,test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

a_tr = sc.fit_transform(a_tr)
a_tt = sc.transform(a_tt)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)
logreg.fit(a_tr,b_tr)

b_predict = logreg.predict(a_tt)

print(b_predict)
print(b_tt)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(b_predict, b_tt)

print(cm)
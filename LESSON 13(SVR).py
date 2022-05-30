#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:15:21 2022

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

from sklearn.svm import SVC
svc = SVC(kernel="linear") #it can be linear,rbf,polynomial....
svc.fit(a_tr,b_tr)

b_predict = svc.predict(a_tt)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(b_tt,b_predict)
print("SVC")
print(cm)
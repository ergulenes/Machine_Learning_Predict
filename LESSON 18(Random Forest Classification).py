#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:12:07 2022

@author: enes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plst

datas = pd.read_csv("datas.csv")

a = datas.iloc[:,1:4].values
b = datas.iloc[:,4:].values
print(b)


from sklearn.model_selection import train_test_split

a_tr, a_tt, b_tr, b_tt = train_test_split(a,b,test_size=0.33, random_state=0)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")

rfc.fit(a_tr,b_tr)
b_predict = rfc.predict(a_tt)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(b_tt,b_predict)
print("RFC")
print(cm)
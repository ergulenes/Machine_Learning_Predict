#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:49:32 2022

@author: enes
"""
#Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rev = pd.read_csv("rev.csv")


a = rev.iloc[:,1:2]
b= rev.iloc[:,2:]
A = a.values
B = b.values

from sklearn.preprocessing import StandardScaler

scale1=StandardScaler()
x_train = scale1.fit_transform(x_train)

sclae2=StandardScaler()
x_test = scale1.fit_transform(x_test)

from sklearn.svm import SVR

reg_svr = SVR(kernel="rbf")
reg_svr.fit(AA,BB)

plt.scatter(AA,BB)
plt.plot(AA, reg_svr.predict(AA))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:24:18 2022

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

from sklearn.ensemble import RandomForestRegressor
ranfor = RandomForestRegressor(n_estimators=10, random_state=0)
ranfor.fit(A,B.ravel())

plt.scatter(A,B)
plt.plot(A, ranfor.predict(A))

from sklearn.metrics import r2_score
print("Value of R2 for Random Forest")
print(r2_score(B,ranfor.predict(A)))
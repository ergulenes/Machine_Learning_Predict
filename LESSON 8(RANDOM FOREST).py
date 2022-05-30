#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:43:19 2022

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

plt.plot(A,B)
plt.scatter([[6.6]], ranfor.predict([[6.6]]))

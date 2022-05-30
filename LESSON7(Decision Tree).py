#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 18:49:25 2022

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

from sklearn.tree import DecisionTreeRegressor
a_dt = DecisionTreeRegressor(random_state=0)
a_dt.fit(A,B)

plt.scatter(A, B)
plt.plot(A, a_dt.predict(A))

print(a_dt.predict([[13]]))
print(a_dt.predict([[6.6]]))

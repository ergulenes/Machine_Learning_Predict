#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 21:36:15 2022

@author: enes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

costumer = pd.read_csv("musteriler.csv")

a = costumer.iloc[:,3:].values

from sklearn.cluster import KMeans

km = KMeans (n_clusters=3, init="k-means++")
km.fit(a)
print(type(km))

print(km.cluster_centers_)

sonuclar = []
for i in range(1,11):   
    km = KMeans (n_clusters=i, init="k-means++", random_state=(123)) 
    km.fit(a)
    sonuclar.append(km.inertia_)

plt.plot(range(1,11),sonuclar)
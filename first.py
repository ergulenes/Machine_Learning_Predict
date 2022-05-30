
#Lesson_6
#Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###CODES

#Uploading Data

datas = pd.read_csv("datas.csv")

height = datas[["boy"]]

heightweight = datas[["boy","kilo"]]

print(datas)

print(heightweight)

print(height)


sales = pd.read_csv("sales.csv")
print(sales)

mounth = sales[["Aylar"]]

revenue = sales[["Satislar"]]

#iloc ile verileri ayrıştırma
revenue2 = sales.iloc[:,:1].values
print("aaaaaaaaaa")
print(revenue2)

#Diveded train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(mounth,revenue,test_size=0.33, random_state=0)

#building model(linear regression)
from sklearn.linear_model import LinearRegression 
lr=LinearRegression()
print(type(lr))
print(type(lr.fit(x_train,y_train)))

est=lr.predict(x_test)

#sıralama
x_train = x_train.sort_index()
y_train = y_train.sort_index()

#Grafikleştirme
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))


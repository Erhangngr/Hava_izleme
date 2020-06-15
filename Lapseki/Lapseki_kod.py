# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sea
data = pd.read_excel('Lapseki.xlsx')

data['PM10'] = data['PM10'].astype('float')
data['SO2'] = data['SO2'].astype('float')
data['NO2'] = data['NO2'].astype('float')
data['O3'] = data['O3'].astype('float')
data['Sebze'] = data['Sebze'].astype('float')
print(data.mean())

data = data.fillna(data.mean())


plt.figure(figsize=(12,6))
plt.plot(data.Tarih,data.NO2)
plt.show()


plt.figure(figsize=(24,12))
plt.subplot(2,2,1)
plt.plot(data.Tarih,data.PM10,color="red")
plt.plot(data.Tarih,data.SO2,color="blue")
plt.plot(data.Tarih,data.NO2,color="black")
plt.plot(data.Tarih,data.O3,color="gray")
plt.xlabel("Tarih")
plt.ylabel("PM10-SO2-NO2-NO-O3 Miktarı")
plt.title("Zamana Göre PM10(Red)-SO2(Blue)-NO2(Black)-NO(Orange)-O3(Gray) değişimi")
plt.show()



sea.scatterplot(x ="SO2",y="NO2", data=data)
plt.show()
sea.scatterplot(x ="PM10",y="O3", data=data,color="red")
plt.show()


veri=pd.read_excel("Lapseki.xlsx",index_col=0)
veri.plot.bar(stacked=True,figsize=(50,20))


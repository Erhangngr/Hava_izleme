# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:36:06 2020

@author: iamer
"""

# -*- coding: utf-8 -*-

import pandas as pd

data=pd.read_excel("Çanakkale_Merkez.xlsx")

MaxPM10=(max(data['PM10']))
MaxSO2=(max(data['SO2']))

MaxNO2=(max(data['NO2']))
MaxNO=(max(data['NO']))
MaxO3=(max(data['O3']))

MinPM10=(min(data['PM10']))
MinSO2=(min(data['SO2']))

MinNO2=(min(data['NO2']))
MinNO=(min(data['NO']))
MinO3=(min(data['O3']))


data = data.fillna(data.mean())

MaxTarih=[]
MinTarih=[]
say=0

for i in data.PM10:
    
    if i==MaxPM10:
        MaxTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
            
for i in data.SO2:
    
    if i==MaxSO2:
        MaxTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    

for i in data.NO2:
    
    if i==MaxNO2:
        MaxTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    
for i in data.NO:
    
    if i==MaxNO:
        MaxTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    
for i in data.O3:
    
    if i==MaxO3:
        MaxTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    
for i in data.PM10:
    
    if i==MinPM10:
        MinTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    
for i in data.SO2:
    
    if i==MinSO2:
        MinTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1


for i in data.NO2:
    
    if i==MinNO2:
        MinTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    
for i in data.NO:
    
    if i==MinNO:
        MinTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    
for i in data.O3:
    
    if i==MinO3:
        MinTarih.append(data.Tarih[say])
        say=0
        break
    say=say+1
    
print(MaxTarih,"\n") 
print("\n",MinTarih)



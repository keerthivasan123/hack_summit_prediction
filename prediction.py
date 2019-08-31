import warnings
import tkinter as tk

import itertools
import numpy as np
import sqlite3 as ap
warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm
import datetime
import calendar
import sys
f = open('nul', 'w')
sys.stderr = f
def getPrediction(productCode, inputDate):
    sampleInterval = "W"
    dow = datetime.datetime.strptime(inputDate, '%Y-%m-%d').weekday()
    sampleInterval = "W-" + ((calendar.day_name[dow][0:3]).upper())
    from sklearn import neighbors
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    cnx=ap.connect(r'C:\Users\keerthivasan\Desktop\hack_summit_prediction\inventory.db')
    df=pd.read_sql_query("Select * from outbound",cnx)
    df.IssueDate=df.IssueDate.astype('datetime64')
    a=df.loc[df['Id'] == 'P2002']
    a = a.groupby('IssueDate')['Quantity'].sum().reset_index()
    a = a.set_index('IssueDate')
    y = a['Quantity'].resample(sampleInterval).mean()
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    pred_uc = results.get_forecast(steps=200)
    pred_ci = pred_uc.conf_int()
    d = pred_uc.predicted_mean
    f = d.to_dict()
    newd = {}
    for k,v in f.items():
        newd[k.to_pydatetime().strftime('%Y-%m-%d')] = int(v)
    return (newd[inputDate])

if __name__ == "__main__":
    productCode = "P2002"
    inputDate = "2020-11-03"
    print(inputDate)
    splited_date=inputDate.split('-')
    list=[]
    for x in range(10):
        send_date=''
        date=int(splited_date[2])+x
        send_date+=splited_date[0]
        send_date+='-'
        send_date+=splited_date[1]
        send_date+='-'
        if(date<10):
            send_date+='0'
        send_date+= str(date)
        list.append(getPrediction(productCode, send_date))
    print(list)

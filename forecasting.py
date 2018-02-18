# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 22:24:17 2017

@author: Fuyan
"""

import csv
import pandas as pd
import numpy as np
import re
from datetime import datetime
import datetime as dtmodule
from matplotlib import pyplot as plt
from fbprophet import Prophet as prophet
import logging
import pyflux as pf
import random
import xgboost as xgb
from sklearn import *


#File folder path
inFoldPath="C:\\Users\\Fuyan\\Desktop\\Recruit Restaurant Visitor Forecasting\\originalData\\"
outFoldPath="C:\\Users\\Fuyan\\Desktop\\Recruit Restaurant Visitor Forecasting\\addressedData\\"
predictPath="C:\\Users\\Fuyan\\Desktop\\Recruit Restaurant Visitor Forecasting\\prediction\\"



""" All this part is for time series method

# Calculate the days differece between two date string
def days_between(d1, d2="2016-01-01"):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)



#Define a split function to split datetime
def dateTimeSplit(datetime):
    a=datetime.split(" ")
    return (a[0],a[1])

#Files to load 
files=["air_store_info","air_reserve","air_visit_data","date_info","hpg_reserve","hpg_store_info","store_id_relation","sample_submission"]

#Store files in a dictionary object
fileDic={}
for f in files:
    fileDic[f]=pd.read_csv(inFoldPath+f+".csv")

#Store store relation as dictionary 
storeRel={}
for store in fileDic["store_id_relation"]:    
    store=store[0].split(",")
    storeRel[store[1]]=store[0]
    
#Store store_info in air as a dictionary
air_store_info={}
for store in fileDic["air_store_info"]:
    store=store[0].split(",")
    air_store_info[store[0]]=store[1:]
    

#Split reserves in hpg into those in air and others 
hpg_reserve_in_air=[]
hpg_reserve_only=[]
for store in fileDic["hpg_reserve"]:
    store=store[0].split(",")
    if not store[0] in storeRel:
        hpg_reserve_only.append(store)
    else:
        store[0]=storeRel[store[0]]
        hpg_reserve_in_air.append(store)


#Define store_record to store every restaurant records
#{id:timerecords,...}
    #time records={date:[visit number, reserve number, reserve time detail],...}
        #reserve time detail=[record,...]
            #record=[visit time, reserve time ,number]

store_record={}

#Load visit data into store_record
for store in fileDic["air_visit_data"]:
    store=store[0].split(",")
    if store[0] in store_record:
        date=store[1]
        visitNum=int(store[2])
        store_record[store[0]][date]=[visitNum,0,[]]
    else:
        date=store[1]
        visitNum=int(store[2])
        store_record[store[0]]={}
        store_record[store[0]][date]=[visitNum,0,[]]
        
#load reserve data into store_record

for store in fileDic["air_reserve"]:
    store=store[0].split(",")
    if store[0] in store_record:
        (date,time)=dateTimeSplit(store[1])
        if date in store_record[store[0]]:
            num=int(store[3])
            record=store[1:]
            store_record[store[0]][date][1]+=num
            store_record[store[0]][date][2].append(record)
        else:
#            print("reserve but not visit on date: %s"%date)
            pass
#        print("%s is not in visit records"%store[0])





#Addressed timeseries data for stores



def day_visit(store):
    daylabel=[]
    visit=[]
    for day in store:
        daylabel.append(days_between(day))
        visit.append(store[day][0])
    return (daylabel,visit)


       
#Plot summary week figure
def plotSumWeek(daylabels,visits,ii):
    week=[0]*7
    for i in range(len(daylabels)):
        week[(daylabels[i]+4)%7]+=visits[i]
    plt.figure()
    plt.plot([i for i in range(7)],week,'go')
    print("plotSumWeek is past")
    plt.savefig('C:\\Users\\Fuyan\\Desktop\\Recruit Restaurant Visitor Forecasting\\Figures\\week'+str(ii)+'.png')
    plt.show()
    return 

#Plot one specific month figure
    
def plotOneMonth(daylabels,visits,date):
    beginday=days_between(date)
    beginindex=0
    for i in range(len(daylabels)):
        if daylabels[i]>=beginday:
            beginindex=i
            break
    monthvisit=visits[beginindex:beginindex+30]
    print("beginindex is %d  and visit length is %d"%(beginindex,len(visits)))
    month=daylabels[beginindex:beginindex+30]
    plt.figure()
    plt.plot(month,monthvisit,'r-')
    print("plotOneMonth is past")
    plt.show()
    return
    

#Plot average visit number perday of each month
def plotMonthes(daylabels,visits):
    splitdays=[("2016-%d-01"%i) for i in range(1,13)]+[("2017-%d-01"%i) for i in range(1,5)]
    splitdays=[days_between(day) for day in splitdays]
    splitdays+=[10000]
    month=[]
    j=0
    monthvisit=0
    num=0
    for i in range(len(daylabels)):
        if daylabels[i]>=splitdays[j+1]:
            while not (daylabels[i]<splitdays[j+1] and daylabels[i]>=splitdays[j]):                
                j+=1
                month.append(monthvisit/num if not num==0 else 0)
                monthvisit=0
                num=0
            
            monthvisit=visits[i]
            num=1
        else:
            monthvisit+=visits[i]
            num+=1
    if not num==0:
        month.append(monthvisit/num)
    plt.figure()
    plt.plot([i for i in range(len(month))],month,"bo")
    print("plotMonthes is past")
    plt.show()
    return

# Plot different week day
def plotWeekDay(daylabels,visits,weekday):
    week=[]
    weeklabel=[]
    for i in range(len(daylabels)):
        if (daylabels[i]+4)%7==weekday:
            week.append(visits[i])
            weeklabel.append(int(daylabels[i]/7))
    plt.figure()
    color=['ro','ko','yo','go','co','bo','mo']
    plt.plot(weeklabel,week,color[weekday])
    plt.show()
    
#transform date dictionary into fbprophet dateframe
def prophetData(store):
    ds=[]
    y=[]
    for day in store:
        ds.append(day)
        y.append(store[day][0])
    return pd.DataFrame({'ds':ds,'y':y})

def prophetSplit(store):
    trainds=[]
    trainy=[]
    test={}
    l=len(store)
    i=-1
    for day in store:
        i+=1
        if i>=l-39:
            test[day]=store[day][0]
        else:
            trainds.append(day)
            trainy.append(store[day][0])
    return (pd.DataFrame({'ds':trainds,'y':trainy}),test)
            


# Fill missing date data as zero if date is within active period
def fillMissDate(store):
    a=list(store.keys())
    beginday=a[0]
    lastday="2017-04-22"
    a=0
    filldays={}
    firstday=datetime.strptime(beginday, "%Y-%m-%d")
    for i in range(days_between(lastday,beginday)+1):
        day=(firstday+dtmodule.timedelta(i)).strftime('%Y-%m-%d')
        if day in store:
            filldays[day]=store[day]
        else:
            filldays[day]=[0,0,[]]
    return filldays
           
def visitmean(store):
    total=0
    for day in store:
        total+=store[day][0]
    return total/len(store)
        
# Input is the timeseries data for each store
# Output is the seven dataframe store data in each weekday
def dataInWeek(store):
    #Define a dictionary to store data and index for each weekday
    result={}
    for i in range(7):
        result['t'+str(i)]=[]
        result['y'+str(i)]=[]
    for day in store:
        days=(days_between(day)+4)
        w=days%7
        index=int(days/7)
        result['t'+str(w)].append(index)
        result['y'+str(w)].append(store[day][0])
    return result


#Split store time data into train and test set splitted by 
    #2017-03-12. All data structure keeps invariant
def storeSplit(store):
    train={}
    test={}
    for day in store:
        if days_between(day)<=days_between("2017-03-12"):
            train[day]=store[day]
        else:
            test[day]=store[day]
    return (train,test)


holidays=pd.DataFrame({
  'holiday': 'goldenweek',
  'ds': pd.to_datetime(['2016-04-29', '2016-04-30', '2016-05-01',
                        '2016-05-02', '2016-05-03', '2016-05-04',
                        '2016-05-05','2017-04-29', '2017-04-30', 
                        '2017-05-01', '2017-05-02', '2017-05-03', 
                        '2017-05-04', '2017-05-05' ]),
  'lower_window': 0,
  'upper_window': 1,
})
    
    
help(prophet.__init__)
i=0
result=[]
for store in store_record:
    if i<117:
        i+=1
        continue
    elif i>127:
        break
    store=store_record[store]
    mea=visitmean(store)
    store=fillMissDate(store)
    train,test=prophetSplit(store)
    m=prophet(changepoint_prior_scale=0.2,seasonality_prior_scale=10.0)
    train['cap']=mea
    m.fit(train)
    
    future = m.make_future_dataframe(periods=39)  
    future['cap']=mea      
    forecast = m.predict(future)
    predictValue=forecast[['ds', 'yhat']]
    predictValue=predictValue.iloc[-39:]
    for index, row in predictValue.iterrows():
        predicty=row['yhat'] if row['yhat']>0 else 0
        testy=test[row['ds'].strftime('%Y-%m-%d')]
        print(predicty,testy)
        result.append((np.log(1+predicty)-np.log(1+testy))**2)    
    
    i+=1
result=np.sqrt(sum(result)/len(result))  
print("final score is :%f"%result)    


#def the crossjoin of two dateframe

def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res
    
#Naive fbprophet forecast :0.538
    
#add monthly seasonality: 0.560
    

    
#add missing date as zero otherwise it will be filled as median
    




#adjust parameters to fit model with test set


#def new forecast model by predicting different weekdays seperately

#result is not very good


End of method for time series
"""


#data preaddressing with pandas dataFrame

files=["air_store_info","air_reserve","air_visit_data","date_info","hpg_reserve","hpg_store_info","store_id_relation","sample_submission"]

#Store files in a dictionary object
fileDic={}
for f in files:
    fileDic[f]=pd.read_csv(inFoldPath+f+".csv")
    
    
    
# Reservation preaddressing
    
air_reserve=fileDic['air_reserve']
hpg_reserve=fileDic['hpg_reserve']
store_id_relation=fileDic['store_id_relation']

air_reserve['visit_date']=pd.to_datetime(air_reserve['visit_datetime']).dt.date
air_reserve=air_reserve.groupby(by=['air_store_id','visit_date'],as_index=False)['reserve_visitors'].sum().rename(
        columns={'reserve_visitors':'air_visitors'})

hpg_reserve['visit_date']=pd.to_datetime(hpg_reserve['visit_datetime']).dt.date
hpg_reserve=hpg_reserve.groupby(by=['hpg_store_id','visit_date'],as_index=False)['reserve_visitors'].sum().rename(
        columns={'reserve_visitors':'hpg_visitors'})


air_reserve=air_reserve.merge(store_id_relation,how='outer',on=['air_store_id'])
reserve=air_reserve.merge(hpg_reserve,how='left',on=['hpg_store_id','visit_date'])

reserve['reserve_visitors']=reserve['air_visitors'].fillna(0)+reserve['hpg_visitors'].fillna(0)
reserve=reserve[['air_store_id','visit_date','reserve_visitors']]

del air_reserve
del hpg_reserve
del store_id_relation
reserve=reserve[reserve.reserve_visitors>0]


# end resrvation preaddressing with reserve left for further usage


submission=fileDic['sample_submission']
submission["air_store_id"]=submission.apply(lambda x: re.sub("\_\d\d\d\d-\d\d-\d\d","",x['id']),axis=1 )
submission['visit_date']=submission.apply(lambda x: x['id'].split('_')[2],axis=1 )
submission.drop(['id'],axis=1,inplace=True)

predict=submission
predict['visitors']=None


air_visit=fileDic["air_visit_data"]
date_info=fileDic["date_info"]
air_store_info=fileDic["air_store_info"]




air_visit=pd.concat([air_visit,predict])
air_visit["visit_date"]=pd.to_datetime(air_visit["visit_date"])
air_visit.index=range(len(air_visit["visit_date"]))



air_visit=air_visit.merge(air_store_info,how="left",on="air_store_id")



date_info=date_info.rename(columns={"calendar_date":'cld',"day_of_week":'dow',"holiday_flg":"hflg"})

date_info["cld"]=pd.to_datetime(date_info['cld'])

date_info=date_info.rename(columns={'cld':"visit_date"})

date_info[date_info.hflg==1]
print(air_visit.shape)

air_visit=air_visit.merge(date_info,how="left",on="visit_date")

air_visit["year"]=air_visit["visit_date"].dt.year
air_visit['dow']=air_visit["visit_date"].dt.dayofweek
air_visit["month"]=air_visit["visit_date"].dt.month


air_visit[air_visit.hflg==1]


train=air_visit.query('visitors >=0')

train=train.query('month==2 or month==3 or month==4 or month==5')

test=air_visit[air_visit.visitors.isnull()]



train1=train[train.visit_date.dt.date<dtmodule.date(2017,3,1)]



cross=train[train.visit_date.dt.date>=dtmodule.date(2017,3,1)]

cross1=cross[cross.visit_date.dt.date<dtmodule.date(2017,3,23)]

cross2=cross[cross.visit_date.dt.date>=dtmodule.date(2017,3,23)]



   






train1['visitors']=train1['visitors'].apply(pd.to_numeric)

# drop holiday flag which is not goldenweek, because different holiday may 
# have different effect on visitors

train1_noholiday=train1[train1.hflg==0]
train1_16goldenweek=train1[train1.visit_date.dt.date>=dtmodule.date(2016,4,29)]
train1_16goldenweek=train1_16goldenweek[dtmodule.date(2016,5,5)>=train1_16goldenweek.visit_date.dt.date]
train1_16goldenweek['hflg']=[1]*len(train1_16goldenweek['hflg'])


train1=pd.concat([train1_noholiday,train1_16goldenweek])

cross1=cross1[cross1.hflg==0]

# define the min, max, mean, median, standard deviation for stores with weekdays

tem=train1.groupby(['air_store_id','dow'],as_index=False)["visitors"].min().rename(
        columns={"visitors":"visitors_min"})
train1=train1.merge(tem,how='left',on=['air_store_id','dow'])
cross1=cross1.merge(tem,how='left',on=['air_store_id','dow'])

tem=train1.groupby(['air_store_id','dow'],as_index=False)["visitors"].max().rename(
        columns={"visitors":"visitors_max"})
train1=train1.merge(tem,how='left',on=['air_store_id','dow'])
cross1=cross1.merge(tem,how='left',on=['air_store_id','dow'])

tem=train1.groupby(['air_store_id','dow'],as_index=False)["visitors"].mean().rename(
        columns={"visitors":"visitors_mean"})
train1=train1.merge(tem,how='left',on=['air_store_id','dow'])
cross1=cross1.merge(tem,how='left',on=['air_store_id','dow'])

tem=train1.groupby(['air_store_id','dow'],as_index=False)["visitors"].median().rename(
        columns={"visitors":"visitors_median"})
train1=train1.merge(tem,how='left',on=['air_store_id','dow'])
cross1=cross1.merge(tem,how='left',on=['air_store_id','dow'])

tem=train1.groupby(['air_store_id','dow'],as_index=False)["visitors"].agg(np.std,ddof=1).rename(
        columns={"visitors":"visitors_std"})
train1=train1.merge(tem,how='left',on=['air_store_id','dow'])
cross1=cross1.merge(tem,how='left',on=['air_store_id','dow'])

# define the monthly daily average for stores with months and drop the first month


def newDate(x,y):
    return dtmodule.date(x,y,1)+pd.to_timedelta('32 days')


tem1=train1.groupby(by=['air_store_id','year','month'],as_index=False)['visitors'].mean().rename(
        columns={'visitors':'bemonav'})
tem2=train1.groupby(by=['air_store_id','year','month'],as_index=False)['visitors'].count().rename(
        columns={'visitors':'bemoncount'})

tem=tem1.merge(tem2,how='left',on=['air_store_id','year','month'])

tem["newdate"]=tem.apply(lambda x: newDate(x['year'], x['month']), axis=1)

tem['newdate']=pd.to_datetime(tem['newdate'])
tem['year']=tem['newdate'].dt.year
tem['month']=tem['newdate'].dt.month

tem=tem.drop(['newdate'],axis=1)

train1=train1.merge(tem,how='left',on=['air_store_id','year','month'])

cross1=cross1.merge(tem,how='left',on=['air_store_id','year','month'])



# Count the number and average of each weekday last month


tem1=train1.groupby(by=['air_store_id','dow','year','month'],as_index=False)['visitors'].mean().rename(
        columns={'visitors':'bemonweekav'})
tem2=train1.groupby(by=['air_store_id','dow','year','month'],as_index=False)['visitors'].count().rename(
        columns={'visitors':'bemonweekcount'})
tem=tem1.merge(tem2,how='left',on=['air_store_id','dow','year','month'])

tem["newdate"]=tem.apply(lambda x: newDate(x['year'], x['month']), axis=1)

tem['newdate']=pd.to_datetime(tem['newdate'])
tem['year']=tem['newdate'].dt.year
tem['month']=tem['newdate'].dt.month

tem=tem.drop(['newdate'],axis=1)

train1=train1.merge(tem,how='left',on=['air_store_id','dow','year','month'])

cross1=cross1.merge(tem,how='left',on=['air_store_id','dow','year','month'])


#drop first month which has nan for bemonav

train1=train1.dropna(subset=['bemonav'])


#preaddress train  and test data

train['visitors']=train['visitors'].apply(pd.to_numeric)

# drop holiday flag which is not goldenweek, because different holiday may 
# have different effect on visitors
bedate=dtmodule.date(2016,4,29)
endate=dtmodule.date(2016,5,5)
train_noholiday=train[train.hflg==0]
train_noholiday1=train_noholiday[train_noholiday.visit_date.dt.date<bedate]
train_noholiday2=train_noholiday[train_noholiday.visit_date.dt.date>endate]

train_16goldenweek=train[train.visit_date.dt.date>=bedate]
train_16goldenweek=train_16goldenweek[endate>=train_16goldenweek.visit_date.dt.date]
train_16goldenweek['hflg']=[1]*len(train_16goldenweek['hflg'])
train=pd.concat([train_noholiday1,train_noholiday2,train_16goldenweek])

# drop holiday flag which is not goldenweek, because different holiday may 
# have different effect on visitors

bedate=dtmodule.date(2017,4,29)
endate=dtmodule.date(2017,5,5)
test_noholiday1=test[test.visit_date.dt.date<bedate]
test_noholiday2=test[test.visit_date.dt.date>endate]
test_17goldenweek=test[test.visit_date.dt.date>=bedate]
test_17goldenweek=test_17goldenweek[endate>=test_17goldenweek.visit_date.dt.date]
test_17goldenweek['hflg']=[1]*len(test_17goldenweek['hflg'])
test=pd.concat([test_noholiday1,test_noholiday2,test_17goldenweek])

print(test.shape)

print(test)
# define the monthly daily average for stores with months and drop the first month

tem1=train.groupby(by=['air_store_id','year','month'],as_index=False)['visitors'].mean().rename(
        columns={'visitors':'bemonav'})
tem2=train.groupby(by=['air_store_id','year','month'],as_index=False)['visitors'].count().rename(
        columns={'visitors':'bemoncount'})
tem=tem1.merge(tem2,how='left',on=['air_store_id','year','month'])

tem["newdate"]=tem.apply(lambda x: newDate(x['year'], x['month']), axis=1)

tem['newdate']=pd.to_datetime(tem['newdate'])
tem['year']=tem['newdate'].dt.year
tem['month']=tem['newdate'].dt.month

tem=tem.drop(['newdate'],axis=1)

train=train.merge(tem,how='left',on=['air_store_id','year','month'])

test=test.merge(tem,how='left',on=['air_store_id','year','month'])


# Count the number and average of each weekday last month


tem1=train.groupby(by=['air_store_id','dow','year','month'],as_index=False)['visitors'].mean().rename(
        columns={'visitors':'bemonweekav'})
tem2=train.groupby(by=['air_store_id','dow','year','month'],as_index=False)['visitors'].count().rename(
        columns={'visitors':'bemonweekcount'})
tem=tem1.merge(tem2,how='left',on=['air_store_id','dow','year','month'])

print(tem)
tem["newdate"]=tem.apply(lambda x: newDate(x['year'], x['month']), axis=1)

tem['newdate']=pd.to_datetime(tem['newdate'])
tem['year']=tem['newdate'].dt.year
tem['month']=tem['newdate'].dt.month

tem=tem.drop(['newdate'],axis=1)

train=train.merge(tem,how='left',on=['air_store_id','dow','year','month'])

test=test.merge(tem,how='left',on=['air_store_id','dow','year','month'])
#drop first month which has nan for bemonav

train=train.dropna(subset=['bemonav'])



# define the min, max, mean, median, standard deviation for stores with weekdays

tem=train.groupby(['air_store_id','dow'],as_index=False)["visitors"].min().rename(
        columns={"visitors":"visitors_min"})
train=train.merge(tem,how='left',on=['air_store_id','dow'])
test=test.merge(tem,how='left',on=['air_store_id','dow'])

tem=train.groupby(['air_store_id','dow'],as_index=False)["visitors"].max().rename(
        columns={"visitors":"visitors_max"})
train=train.merge(tem,how='left',on=['air_store_id','dow'])
test=test.merge(tem,how='left',on=['air_store_id','dow'])

tem=train.groupby(['air_store_id','dow'],as_index=False)["visitors"].mean().rename(
        columns={"visitors":"visitors_mean"})
train=train.merge(tem,how='left',on=['air_store_id','dow'])
test=test.merge(tem,how='left',on=['air_store_id','dow'])

tem=train.groupby(['air_store_id','dow'],as_index=False)["visitors"].median().rename(
        columns={"visitors":"visitors_median"})
train=train.merge(tem,how='left',on=['air_store_id','dow'])
test=test.merge(tem,how='left',on=['air_store_id','dow'])

tem=train.groupby(['air_store_id','dow'],as_index=False)["visitors"].agg(np.std,ddof=1).rename(
        columns={"visitors":"visitors_std"})
train=train.merge(tem,how='left',on=['air_store_id','dow'])
test=test.merge(tem,how='left',on=['air_store_id','dow'])




#get the information for gendr and area with holiday flag stat

train['visitors']=train['visitors'].apply(pd.to_numeric)


gahtrain=train

gahtrain_noholiday=gahtrain[gahtrain.hflg==0]
gahtrain_16goldenweek=gahtrain[gahtrain.visit_date.dt.date>=dtmodule.date(2016,4,29)]
gahtrain_16goldenweek=gahtrain_16goldenweek[dtmodule.date(2016,5,5)>=gahtrain_16goldenweek.visit_date.dt.date]
gahtrain_16goldenweek['hflg']=[1]*len(gahtrain_16goldenweek['hflg'])


gahtrain=pd.concat([gahtrain_noholiday,gahtrain_16goldenweek])

genre_area_hflg=gahtrain.groupby(by=['air_genre_name','air_area_name','hflg'],as_index=False)['visitors'].mean().rename(
        columns={'visitors':'gah'})

genre_area_nonh=genre_area_hflg[genre_area_hflg.hflg==0].rename(columns={"gah":'ganh'})
genre_area_h=genre_area_hflg[genre_area_hflg.hflg>0]

genre_area_nonh.drop(['hflg'],axis=1,inplace=True)
genre_area_h.drop(['hflg'],axis=1,inplace=True)

print(genre_area_nonh)
genre_area=genre_area_nonh.merge(genre_area_h,how='left',on=['air_genre_name','air_area_name'])

print(genre_area)
genre_area['ngah']=genre_area.apply(lambda x: (x['gah']-x['ganh'])/x['ganh'] if not x['gah']==None else 0, axis=1)
# check different parameters

genre_area['ngah']=genre_area['ngah'].fillna(0)

genre_area.drop(['gah','ganh'],axis=1,inplace=True)

genre_area['hflg']=[1]*len(genre_area['ngah'])

# put genre and area with goldenweek information into train1 and cross

train1=train1.merge(genre_area,how='left',on=['air_genre_name','air_area_name','hflg'])
cross1=cross1.merge(genre_area,how='left',on=['air_genre_name','air_area_name','hflg'])
cross2=cross2.merge(genre_area,how='left',on=['air_genre_name','air_area_name','hflg'])

print(train1.shape,cross1.shape,cross2.shape)

# put genre and area with goldenweek information into train and test

train=train.merge(genre_area,how='left',on=['air_genre_name','air_area_name','hflg'])
test=test.merge(genre_area,how='left',on=['air_genre_name','air_area_name','hflg'])

print(train.shape,test.shape)


"""
  End
The genre and area with goldenweek information 

"""



'''
begin new features from kaggle Kernel For train and test

'''

# NEW FEATURES FROM Georgii Vyshnia
# THE GENRE NAME AND AREA NAME ENCODER


# NEW FEATURES FROM JMBULL
# New features about date int and latitude longtitude
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']





'''
end new features from Kaggle Kernel for train and test
'''




'''
begin new features from kaggle Kernel for train1 and cross1

'''

# NEW FEATURES FROM Georgii Vyshnia
# THE GENRE NAME AND AREA NAME ENCODER


# NEW FEATURES FROM JMBULL
# New features about date int and latitude longtitude
train1['date_int'] = train1['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
cross1['date_int'] = cross1['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train1['var_max_lat'] = train1['latitude'].max() - train['latitude']
train1['var_max_long'] = train1['longitude'].max() - train['longitude']
cross1['var_max_lat'] = cross1['latitude'].max() - test['latitude']
cross1['var_max_long'] = cross1['longitude'].max() - test['longitude']

# NEW FEATURES FROM Georgii Vyshnia
train1['lon_plus_lat'] = train1['longitude'] + train1['latitude'] 
cross1['lon_plus_lat'] = cross1['longitude'] + cross1['latitude']




'''
end new features from Kaggle Kernel
'''



"""
Begin the weather feature for train test train1 cross1
"""


weather=pd.read_csv(outFoldPath+"air_store_visit_weather.csv")
weather=weather[['air_store_id', 'calendar_date', 'avg_temperature',
       'high_temperature', 'low_temperature', 'hours_sunlight',
       'precipitation']]
weather["visit_date"]=pd.to_datetime(weather["calendar_date"])
weather.drop(["calendar_date"],axis=1,inplace=True)
weather['visit_date']=weather['visit_date'].dt.date



train['visit_date']=train['visit_date'].dt.date
test['visit_date']=test['visit_date'].dt.date
train1['visit_date']=train1['visit_date'].dt.date
cross1['visit_date']=cross1['visit_date'].dt.date


train=train.merge(weather,how='left',on=['air_store_id','visit_date'])
test=test.merge(weather,how='left',on=['air_store_id','visit_date'])
train1=train1.merge(weather,how='left',on=['air_store_id','visit_date'])
cross1=cross1.merge(weather,how='left',on=['air_store_id','visit_date'])


"""
end the weather feature 
"""



"""
The part for feature selecting and parameters tuning 

"""


#Fill the NaN with -1 and define predictors

train1=train1.fillna(-1)
train=train.fillna(-1)
test=test.fillna(-1)
cross1=cross1.fillna(-1)



drop=['air_store_id', 'visit_date', 'visitors', 'air_genre_name',
       'air_area_name', 'latitude', 'longitude','year','month','dow','bemonweekcount','bemoncount']



col=[c for c in train if c not in drop]

print(train.shape)
print(test.shape)

# metric for different modeling
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

def paraTuning(lr,n,depth):
    #model setting up and parameters tuning
    
    model1=xgb.XGBRegressor(learning_rate=lr, random_state=3, n_estimators=n, subsample=0.8, 
                          colsample_bytree=0.8, max_depth =depth)
    model2 = ensemble.GradientBoostingRegressor(learning_rate=lr, random_state=3, n_estimators=n, subsample=0.8, 
                          max_depth =depth)
    model3 = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=depth+7)
    
    
    model1.fit(train1[col],np.log1p(train1['visitors'].values))
    
    model2.fit(train1[col],np.log1p(train1['visitors'].values))
    
    model3.fit(train1[col],np.log1p(train1['visitors'].values))
    
    
    tra1=model1.predict(train1[col])
    pre1=model1.predict(cross1[col])
    
    tra2=model2.predict(train1[col])
    pre2=model2.predict(cross1[col])
    
    tra3=model3.predict(train1[col])
    pre3=model3.predict(cross1[col])
    
    print("model1 train score:",RMSLE(tra1,np.log1p(train1['visitors'].values)),'cross score:',RMSLE(pre1,np.log1p(cross1['visitors'].values)))
    print("model2 train score:",RMSLE(tra2,np.log1p(train1['visitors'].values)),'cross score:',RMSLE(pre2,np.log1p(cross1['visitors'].values)))
    print("model3 train score:",RMSLE(tra3,np.log1p(train1['visitors'].values)),'cross score:',RMSLE(pre3,np.log1p(cross1['visitors'].values)))
    return(RMSLE(pre1,np.log1p(cross1['visitors'].values)),RMSLE(pre2,np.log1p(cross1['visitors'].values)),RMSLE(pre3,np.log1p(cross1['visitors'].values)))

lrvalues=[0.04]
nvalues=[500,600,700]
depthvalues=[5,6,7,8]
m=[1,1,1]
LR=[0,0,0]
N=[0,0,0]
D=[0,0,0]
loss=[0,0,0]
for lr in lrvalues:
    for n in nvalues:
        for depth in depthvalues:
            print("lr:",lr,'n:',n,'depth/double neibor:',depth)
            (loss[0],loss[1],loss[2])=paraTuning(lr,n,depth)
            for i in range(3):
                if loss[i]<m[i]:
                    m[i]=loss[i]
                    LR[i],N[i],D[i]=(lr,n,depth)
for i in range(3):
    print("best model",i+1,"   loss:",m[i],"  lr: ",LR[i],"  n: ",N[i], "  depth:", D[i] )

# training and submission preparation
            
"""
The part for submission training  

"""

#Model1     
model=xgb.XGBRegressor(learning_rate=0.05, random_state=3, n_estimators=600, subsample=0.8, 
                          colsample_bytree=0.8, max_depth =5)

model = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=25)


model.fit(train[col],np.log1p(train['visitors'].values))    

result=model.predict(test[col])
result=np.exp(result)-1
test['visitors']=result


#model2

    




"""
Post adjusting result based on closed date and reservation
"""
#reservation adjustment 


test=test.merge(reserve,how='left',on=['air_store_id','visit_date'])
test[test.reserve_visitors.notnull()]
test['reserve_visitors']=test['reserve_visitors'].fillna(0)

test['visitors']=test.apply(lambda x: x['visitors'] if x['visitors']>1.25*x['reserve_visitors'] else 1.25*x['reserve_visitors'],axis=1)

#closed date adjustment

# 100% closed date around 2000 date
test['visitors']=test.apply(lambda x: 1 if x['bemonweekcount']==-1 and x['bemoncount']>20 and x['hflg']==0 and x['visitors']>1 else x['visitors'],axis=1)



test['id']=test.apply(lambda x: x['air_store_id']+'_'+x['visit_date'].strftime('%Y-%m-%d'),axis=1)


predict=test[['id','visitors']]
predict['visitors']=predict.apply(lambda x: x['visitors'] if x['visitors']>0 else 0,axis=1)
predict.to_csv(predictPath+'prediction.csv',index=False)

print(predict)

print(test.shape)

  

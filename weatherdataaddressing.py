# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 00:47:54 2018

@author: Fuyan
"""

from pandas import DataFrame, merge
import pandas as pd
import numpy as np


inPath="C:\\Users\\Fuyan\\Desktop\\Recruit Restaurant Visitor Forecasting\\originalData\\"
outPath="C:\\Users\\Fuyan\\Desktop\\Recruit Restaurant Visitor Forecasting\\addressedData\\"



#Files to load
air_station=pd.read_csv(inPath+"air_store_info_with_nearest_active_station.csv")

files=["air_store_info","air_reserve","air_visit_data","date_info","hpg_reserve","hpg_store_info","store_id_relation","sample_submission"]

fileDic={}
for f in files:
    fileDic[f]=pd.read_csv(inPath+f+".csv")



#create a kronecker product between air_stores/stations with datetime
    
fileDic["date_info"]['keys']=[1]*len(fileDic["date_info"]['calendar_date'])

date_info=fileDic["date_info"][["calendar_date","keys"]]

air_station["keys"]=[1]*len(air_station['air_store_id'])
air_station=air_station[["keys","air_store_id","station_id"]]

air_station_date=merge(air_station,date_info,on="keys")




# get all data for nearby_active_station from station files
nearby=pd.read_csv(inPath+"nearby_active_stations.csv")['id']
nearbyFiles={}

for i in nearby:
    nearbyFiles[i]=pd.read_csv(inPath+"weather\\"+i+".csv")
    

for i in nearbyFiles:
    nearbyFiles[i]['id']=[i]*len(nearbyFiles[i].index)
    
#concate all nearby_active_station data into one single file
    
allnearby=[nearbyFiles[i] for i in nearbyFiles]

nearby_date=pd.concat(allnearby)

nearby_date.index=range(len(nearby_date['id']))


#Fill all missing values based on nearby station data
#Load weather_stations for distance checking
#load feature_manifest for availability of given feature cheking
weather_stations=pd.read_csv(inPath+"weather_stations.csv")
manifest=pd.read_csv(inPath+"feature_manifest"+".csv")


# get longtitude and latitude out of weather stations for distance checking
longtitudelist=list(weather_stations["longitude"].values)
longtitudelist=list(map(float,longtitudelist))
latitudelist=list(weather_stations["latitude"].values)
latitudelist=list(map(float,latitudelist))


# Find the nearest station with given feature available
def nearestStation(station,fea):
    index=weather_stations.index[weather_stations.id==station][0]
    oldla=latitudelist[index]
    oldlon=longtitudelist[index]
    result=None
    distance=100000000000
    for i in range(len(latitudelist)):
        newla=latitudelist[i]
        newlon=longtitudelist[i]
        dis=np.sqrt((oldla-newla)**2+(oldlon-newlon)**2)
        
        if dis<distance and float(manifest.iloc[i][fea])>0.8 and dis>0:
            distance=dis
            result=manifest.iloc[i]['id']
    
    return pd.read_csv(inPath+"weather\\"+result+".csv")

    
    
# Fill the station data by nearest station data
def fillUp(station,fea):
    nearest=nearestStation(station,fea)
    nearbyFiles[station][fea]=nearest[fea]
    return
    
    
    
#Check all missing data and fill it up    
features=['avg_temperature', 'high_temperature',
       'low_temperature','hours_sunlight']

for fea in features:
    print("*****************************")
    temnull=nearby_date[nearby_date[fea].isnull()]
    nearby_date[fea].interpolate(method='nearest',inplace=True)
    for i in temnull["id"].unique():
        a=i
        (sizea,sizeb)=temnull[temnull.id==a].shape
        print(sizea)
        if sizea>400:
            fillUp(i,fea)
 

#Re-concate nearby_active_station data with no missings
    
allnearby=[nearbyFiles[i] for i in nearbyFiles]
nearby_date=pd.concat(allnearby)
nearby_date.index=range(len(nearby_date['id']))




#Final nearby is another file to get the features we want
finalnearby=nearby_date[['id','calendar_date']+features+['precipitation']]

# Fill the precipitation with 0, which is common used in stations
finalnearby['precipitation'].fillna(0,inplace=True)

# Redefine index
finalnearby.index=range(len(finalnearby['id']))


# Prepare air_station_date to merge with weather data
air_station_date['id']=air_station_date['station_id']
air_station_date=air_station_date[['air_store_id','id','calendar_date']]
new_air_station_date=air_station_date.merge(finalnearby,how='left',on=['id','calendar_date'])


# drop station id leave air_store_id and date with weather features
air_store_visit_weather=new_air_station_date[['air_store_id', 'calendar_date', 'avg_temperature',
       'high_temperature', 'low_temperature', 'hours_sunlight',
       'precipitation']]
# Save the final file
air_store_visit_weather.to_csv(outPath+'air_store_visit_weather.csv')

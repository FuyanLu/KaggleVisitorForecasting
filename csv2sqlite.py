# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 18:15:26 2017

@author: Fuyan
"""

import pandas as pd
import sqlite3 as sql


ODataPath="C:\\Users\\Fuyan\\Desktop\\Recruit Restaurant Visitor Forecasting\\originalData\\"
conn=sql.connect(ODataPath+"RestForecast.sqlite")

datafiles=["air_store_info","air_reserve","air_visit_data","date_info","hpg_reserve","hpg_store_info","store_id_relation"]

for f in datafiles:
    df=pd.read_csv(ODataPath+f+".csv")
    
    df.to_sql(name=f,con=conn,if_exists='append', index=False)

conn.close()
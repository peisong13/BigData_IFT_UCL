# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:59:59 2019

@author: Peisong Yang
"""

#%%
# important to set up directories
GITRepoDirectory = "/your/repository/file/here"

# import packages
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime
from SQLQueries import * #if ModuleNotFoundError appears, please switch to current path or run SQLQueries.py manually

#create engine and connect to SQL

engine = create_engine(f"sqlite:///{GITRepoDirectory}/iftcode/DataBases/DBs/DMOBondPrices/BondPrice.db")
con = engine.connect()

#import bond static from SQL
fullBondStatic = pd.read_sql_query(SQLQueryStatic(),engine)

#%%
#check the structure of fullBondStatic
print(fullBondStatic["issue_date"])

#%%
#other information
fullBondStatic.info()
type(fullBondStatic)
fullBondStatic.shape

#%%
# let's pick the element/ISIN in the first row and first column
fullBondStatic.iloc[0,0]

#%%
# if I use the ISIN as argument of the function peviously sourced from the modules
rs = con.execute(SQLReturnBondPrice(fullBondStatic.iloc[0,0]))
tempPrices = pd.DataFrame(rs)
tempPrices.columns = rs.keys()
tempPrices.info()

#%%
# calculate the average as:
averageClean = tempPrices.clean_price.mean()
# or
tempPrices.clean_price.sum()/len(tempPrices.clean_price)
# and:
averageDirty = tempPrices.dirty_price.mean()
#faster:
tempPrices.dirty_price.describe()

#%%
# let's calulate the returns of Clean Prices
# first create two empty columns to insert the returns
tempPrices['clean_ret'] = np.nan
tempPrices['dirty_ret'] = np.nan
tempPrices['dirty_sd'] = np.nan
tempPrices['dirty_avg'] = np.nan

#%%
# calculate return over 5 days
for i in range(4,len(tempPrices.clean_price)):
    tempPrices.loc[i,'clean_ret'] = np.log(tempPrices.loc[i,'clean_price'])-np.log(tempPrices.loc[i-4,'clean_price'])
    tempPrices.loc[i,'dirty_ret'] = np.log(tempPrices.loc[i,'dirty_price'])-np.log(tempPrices.loc[i-4,'dirty_price'])
    
    if i > 24 :
        j = i - 25
        tempPrices.loc[i,'dirty_sd'] = tempPrices.loc[j:i,'dirty_ret'].std()
        tempPrices.loc[i,'dirty_avg'] = tempPrices.loc[j:i,'dirty_ret'].mean()
        
#%%
# check all returns columns have been properly filled
tempPrices.head(30)
tempPrices = tempPrices.loc[5:]
tempPrices.index = range(len(tempPrices))

#%%
# create a new table in SQLite: bond_returns
con.execute(SQLCreateRetTable())

for i in range(len(tempPrices)):
    print('[INFO] Loading into bond_returns',tempPrices.loc[i,'isin_id'],'...\n')
    con.execute(f'INSERT INTO bond_returns (cob_date, clean_ret, dirty_ret, dirty_sd, dirty_avg, isin_ret) \
                VALUES ("{tempPrices.loc[i,"cob_date"]}",\
                {tempPrices.loc[i,"clean_ret"]*100},\
                {tempPrices.loc[i,"dirty_ret"]*100},\
                {("NULL" if np.isnan(tempPrices.loc[i,"dirty_sd"]) else tempPrices.loc[i,"dirty_sd"])},\
                {("NULL" if np.isnan(tempPrices.loc[i,"dirty_avg"]) else tempPrices.loc[i,"dirty_avg"])},\
                "{tempPrices.loc[i,"isin_id"]}")')
#%%
# let's delete the table... we will do some heavy lifting next
con.execute(SQLDeleteTable())

# let's re-create our SQL table by running the query
con.execute(SQLCreateRetTable())

#%%
for isin in fullBondStatic['isin']:
    print('[INFO] ',datetime.datetime.now().strftime('%F %T'),' Extracting from SQLite Bond Prices for ',isin,'...\n')
    rs = con.execute(SQLReturnBondPrice(isin))
    tempPrices = pd.DataFrame(rs)
    tempPrices.columns = rs.keys()
    
    tempPrices['clean_ret'] = np.nan
    tempPrices['dirty_ret'] = np.nan
    tempPrices['dirty_sd'] = np.nan
    tempPrices['dirty_avg'] = np.nan
    
    print('[INFO] Computing returns for ',isin,'...\n')
    
    if len(tempPrices['clean_price']) > 25:
        for i in range(5,len(tempPrices.clean_price)):
            tempPrices.loc[i,'clean_ret'] = np.log(tempPrices.loc[i,'clean_price'])-np.log(tempPrices.loc[i-5,'clean_price'])
            tempPrices.loc[i,'dirty_ret'] = np.log(tempPrices.loc[i,'dirty_price'])-np.log(tempPrices.loc[i-5,'dirty_price'])
        
            if i > 24 :
                j = i - 25
                tempPrices.loc[i,'dirty_sd'] = tempPrices.loc[j:i,'dirty_ret'].std()
                tempPrices.loc[i,'dirty_avg'] = tempPrices.loc[j:i,'dirty_ret'].mean()
    
        tempPrices = tempPrices.loc[5:]
        tempPrices.index = range(len(tempPrices))
    
        print("[INFO] ", datetime.datetime.now().strftime('%F %T')," Insert returns into bond_returns for ", isin,"...\n")
    
        for i in range(len(tempPrices)):
            con.execute(f'INSERT INTO bond_returns (cob_date, clean_ret, dirty_ret, dirty_sd, dirty_avg, isin_ret) \
                        VALUES ("{tempPrices.loc[i,"cob_date"]}",\
                        {tempPrices.loc[i,"clean_ret"]*100},\
                        {tempPrices.loc[i,"dirty_ret"]*100},\
                        {("NULL" if np.isnan(tempPrices.loc[i,"dirty_sd"]) else tempPrices.loc[i,"dirty_sd"])},\
                        {("NULL" if np.isnan(tempPrices.loc[i,"dirty_avg"]) else tempPrices.loc[i,"dirty_avg"])},\
                        "{tempPrices.loc[i,"isin_id"]}")')    
        print("[INFO] ", datetime.datetime.now().strftime('%F %T')," Returns successfully loaded for ", isin,"...\n")
    
    else:
        print("[INFO] ", datetime.datetime.now().strftime('%F %T')," TimeSeries too short. Returns NOT loaded for ", isin,"...\n")

#%%
# switch off connection
con.close()

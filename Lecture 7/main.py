# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:37:03 2019

@author: Peisong Yang
"""

# let's import our Stock Data from MongoDB
import matplotlib.pyplot as plt
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

#%%
# open connectivity to MongoDB
con = MongoClient('mongodb://localhost')
db = con.SP500
col = db.Static

#%%
# MongoDB Query
fullUniverse = col.find({}) # importing all values in R
fullUniverse = pd.DataFrame(fullUniverse)

#%%
# lets see what is in our dataframe
fullUniverse.head()
fullUniverse.info()

#convert type
fullUniverse[['Price','DividendYield','MarketCap','PERatio','PayoutRatio','Beta']] = fullUniverse[['Price','DividendYield','MarketCap','PERatio','PayoutRatio','Beta']].apply(pd.to_numeric,errors='coerce')

AvgMktCap = pd.DataFrame(fullUniverse.groupby('GICSSector')['MarketCap'].mean())

#%%
# cleaning up from nas
cleanedUniverse = fullUniverse.dropna(subset=['PERatio','DividendYield'])

#%%
# running kmeans on Dividend Yield and PERatio
estimator = KMeans(n_clusters=10)
estimator.fit(cleanedUniverse[['DividendYield','PERatio']])

#%%
# visualise data
cleanedUniverse['cluster'] = estimator.labels_
stockCenters = pd.DataFrame(estimator.cluster_centers_)
print(stockCenters)


#%%
# plot results
plt.figure(figsize=(15,7))
plt.scatter(cleanedUniverse.loc[:,'DividendYield'], cleanedUniverse.loc[:,'PERatio'],c=estimator.labels_)
plt.scatter(stockCenters.iloc[:,0],stockCenters.iloc[:,1],c=range(estimator.n_clusters),marker='x')
plt.scatter(stockCenters.iloc[:,0],stockCenters.iloc[:,1],c=range(estimator.n_clusters),alpha=0.3,s=2000)
plt.show()

#%%
# wait look there is an extreme value that is biasing all our work!
# let's get rid of it by applying the Median Absolute Deviation technique
# MAD = median|x - median.x|

PERatioMAD = np.median(np.abs(cleanedUniverse['PERatio'] - np.median(cleanedUniverse['PERatio'])))

# We can calculate the Modified Z-score like this:
MADZScores = (0.6745 * np.abs(cleanedUniverse['PERatio'] - np.median(cleanedUniverse['PERatio'])))/PERatioMAD

# Now we can calculate the score for each point of our sample! 
# As a rule of thumb, we'll use the score of 3.5 as our cut-off value; 
# This means that every point with a score above 3.5 will be considered an outlier.
cleanedUniverseOutliers = cleanedUniverse[MADZScores < 3.5]

#%%
# running kmeans on Dividend Yield and PERatio
estimator = KMeans(n_clusters=10)
estimator.fit(cleanedUniverseOutliers[['PERatio','DividendYield']])

# visualise data
cleanedUniverseOutliers['cluster'] = estimator.labels_
stockCenters = pd.DataFrame(estimator.cluster_centers_)
print(stockCenters)

plt.figure(figsize=(13,6))
plt.scatter(cleanedUniverseOutliers.loc[:,'PERatio'], cleanedUniverseOutliers.loc[:,'DividendYield'],c=estimator.labels_)
plt.scatter(stockCenters.iloc[:,0],stockCenters.iloc[:,1],c=range(estimator.n_clusters),marker='x')
plt.scatter(stockCenters.iloc[:,0],stockCenters.iloc[:,1],c=range(estimator.n_clusters),alpha=0.3,s=2000)
plt.show()

#%%
# Some desc statistics
# and we find the average
# grouping by clusters

stat = pd.DataFrame()
stat['cluster'] = list(range(estimator.n_clusters))
stat['PEMean'] = pd.DataFrame(cleanedUniverseOutliers.groupby('cluster')['PERatio'].mean())
stat['DYMean'] = pd.DataFrame(cleanedUniverseOutliers.groupby('cluster')['DividendYield'].mean())
print(stat)

#%%
# evaluate if we selected the right number of clusters
# A plot of the within groups sum of squares by number of clusters 
# extracted can help determine the appropriate number of clusters.
wss = np.zeros(15)
for i in range(15):
    wss[i] = KMeans(n_clusters=i+1).fit(cleanedUniverseOutliers[['PERatio','DividendYield']]).inertia_
plt.figure(figsize=(13,6))
plt.plot(wss,'co--')

#%%
con.close()

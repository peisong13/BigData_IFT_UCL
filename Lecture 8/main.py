# -*- coding: utf-8 -*-
"""
Created on Sun Dec 4 23:00:03 2019

@author: Peisong Yang
"""

# import packages in current environment

import pandas as pd
import configparser
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA

# important to set up directories
GITRepoDirectory = "your/repo/here"

# loading configurations
config = configparser.ConfigParser()
config.read(GITRepoDirectory + "/iftcode/Scripts/Python/Lecture 8/settings/script.config")

# Import data from csv
housingData = pd.read_csv(GITRepoDirectory + config.get("Directories", "DBFolder") + "/train.csv")

# describe data
housingData.info()
housingData.dtypes

# target
dVariable = "price_doc"

# price is generally a function of sq
iVariable = "life_sq"

# let's check the degree of correlation between our variables
housingData[[dVariable, iVariable]].corr()

# covariance matrix
housingData[[dVariable, iVariable]].cov()

# let's do our first regression:
parameters = housingData[[iVariable, dVariable]].dropna()
model = LinearRegression()
model.fit(parameters[iVariable].values.reshape(-1, 1), parameters[dVariable])

model.intercept_
model.score(parameters[iVariable].values.reshape(-1, 1), parameters[dVariable])
medianResidues = np.median(parameters[dVariable] - model.predict(parameters[iVariable].values.reshape(-1, 1)))
medianResidues

# plot regression
plt.xlabel(iVariable)
plt.ylabel(dVariable)
plt.scatter(parameters[iVariable], parameters[dVariable], marker='o', color='', edgecolors='k')
plt.scatter(parameters[iVariable], np.log(parameters[dVariable]), marker='o', color='', edgecolors='k')

# there are a number of nas, outliers or wrong values that are affecting our estimates
# let's remove all extreme values with the help of a control variable (full_sq)

fullSqSdev = np.std(housingData["full_sq"].dropna(), ddof=1)  # pay attention to the ddof: delta degrees of freedom
for index, row in housingData.iterrows():
    if pd.isnull(row[iVariable]) or row[dVariable] > row[iVariable] + 3 * fullSqSdev or row[dVariable] < row[iVariable] + 3 * fullSqSdev:
        housingData.loc[index, iVariable] = row["full_sq"]

# let's remove that big outliers manually
housingData = housingData[housingData[iVariable] < 1000]
# look at the chart
plt.xlabel(iVariable)
plt.ylabel(dVariable)
plt.scatter(housingData[iVariable], housingData[dVariable], marker='o', color='', edgecolors='k')
# probably we want to remove anything that is too big
housingData = housingData[housingData[iVariable] < 200]
plt.scatter(housingData[iVariable], housingData[dVariable], marker='o', color='', edgecolors='k')
# and anything that is too small
housingData = housingData[(housingData[iVariable] < 200) & (housingData[iVariable] > 6)]
plt.scatter(housingData[iVariable], housingData[dVariable], marker='o', color='', edgecolors='k')

# now our chart makes a bit more sense and we can re-run the analysis in part 1
# let's regress them without outliers

parameters = housingData[[iVariable, dVariable]].dropna()
model = LinearRegression()
model.fit(parameters[iVariable].values.reshape(-1, 1), parameters[dVariable])
model.intercept_
model.score(parameters[iVariable].values.reshape(-1, 1), parameters[dVariable])
medianResidues = np.median(parameters[dVariable] - model.predict(parameters[iVariable].values.reshape(-1, 1)))
medianResidues
residues = parameters[dVariable] - model.predict(parameters[iVariable].values.reshape(-1, 1))
np.corrcoef(residues, housingData[dVariable])
# plot regression
plt.scatter(parameters[iVariable], parameters[dVariable], marker='o', color='', edgecolors='k')
plt.plot(parameters[iVariable], model.predict(parameters[iVariable].values.reshape(-1, 1)))

# still our model is not able to catch properly the pattern
# there is lot of dispersion in particular among flats with sq larger than 1000

# let's look at regional patterns
byAreaRegression = pd.DataFrame()

# there are limitations using sklearn
# it's hard to show p-value and the significance
# using statsmodels would be easier

for subArea in set(housingData["sub_area"]):
    tempHousing = housingData[housingData["sub_area"] == subArea]
    try:
        X = sm.add_constant(tempHousing[iVariable])
        y = tempHousing[dVariable]
        model = sm.OLS(y, X).fit()
        tempSeries = pd.Series()
        tempSeries["subArea"] = subArea
        tempSeries["rsq"] = model.rsquared
        tempSeries["interc"] = model.params[0]
        tempSeries["beta"] = model.params[1]
        tempSeries["is.significant"] = 1 if model.pvalues[1] < 0.01 else 0
        tempSeries["corResIvar"] = np.corrcoef(model.resid, y)[0, 1]
        tempSeries["NoObs"] = len(y)
        byAreaRegression = byAreaRegression.append(tempSeries, ignore_index=True)
    except:
        pass

PSStudy = housingData[housingData["sub_area"] == 'Poselenie Sosenskoe']
rSq = []; multilinearOutput = []; VIF = []; significanceOutput = []; NoSimulations = 300
for i in range(NoSimulations):
    """
    sampleRegression = pd.DataFrame(PSStudy.columns)[:-1].sample(35)[0].to_list()
    # while loop to make sure that "life_sq" variable is not double counted in sample
    while "life_sq" in sampleRegression:
        sampleRegression[sampleRegression.isin(["life_sq"])] = pd.DataFrame(PSStudy.columns)[:-1].sample(1).iloc[0, 0]
    # fatal: re-sampling might get a column that already exists
    sampleRegression.drop_duplicates(inplace=True)
    # also need to add "lift_sq" back
    # try another way to write this
    """
    while True:
        sampleRegression = pd.DataFrame(PSStudy.columns)[:-1].sample(35)[0].to_list()
        if "life_sq" not in sampleRegression:
            break
    sampleRegression.append("life_sq")

    tmpVar = PSStudy[sampleRegression]
    # lets check if we have enough obs to do our analysis
    colToRemove = []
    for column in tmpVar.columns:
        if sum(~tmpVar[column].isna()) < 50:
            colToRemove.append(column)
        elif tmpVar[column].dtypes == "object":
            colToRemove.append(column)

    # remove from sample those columns that do not have enough observations
    if len(colToRemove) != 0:
        tmpVar = tmpVar.drop(colToRemove, axis=1)

    tmpVar[dVariable] = PSStudy[dVariable]
    X = sm.add_constant(tmpVar[tmpVar.columns[:-1]])
    y = tmpVar[tmpVar.columns[-1]]
    model = sm.OLS(y, X, missing='drop').fit()

    rSq.append(model.rsquared)
    multilinearOutput.append([model.summary(), model.params])
    VIF.append(1 / (1 - model.rsquared))
    significanceOutput.append(1 if model.pvalues[1] < 0.01 else 0)

max(rSq)
modelPCA = multilinearOutput[rSq.index(max(rSq))][1].index.to_list()
# PCA Analysis ------------------------------------------------------------

PCADf = PSStudy[modelPCA]
PCADf = PCADf.dropna()
np.corrcoef(PCADf)
# run the principal component ---------------------------------------------
noiceVar = []; expVarRatio = []; score=[]
for i in range(1, 11):
    PCRModel = PCA(n_components=i)
    PCRModel.fit(PCADf[PCADf.columns[:-1]], y=PCADf[PCADf.columns[-1]])
    noiceVar.append(PCRModel.noise_variance_)
    expVarRatio.append(sum(PCRModel.explained_variance_ratio_))
    score.append(PCRModel.score(PCADf[PCADf.columns[:-1]],y=PCADf[PCADf.columns[-1]]))

plt.subplot(221)
plt.xlabel("number of components")
plt.ylabel("estimated noice variance")
plt.plot(range(1, 11), noiceVar)

plt.subplot(222)
plt.xlabel("number of components")
plt.ylabel("total variance explained")
plt.plot(range(1, 11), expVarRatio)

plt.subplot(212)
plt.xlabel("number of components")
plt.ylabel("score")
plt.plot(range(1, 11), score)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
plt.show()

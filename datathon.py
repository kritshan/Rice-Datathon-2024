#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:44:00 2024

@author: Will
"""
"""
import pandas as pd

df = pd.read_csv("/Users/Will/Desktop/datathon/training.csv")
df2 = df.corr()
print(df.isna().sum() / 29692)


df = df.drop(["average_stage_length", "average_proppant_per_stage", "average_frac_fluid_per_stage", "number_of_stages","pad_id","ffs_frac_type"],axis = 1)

print(df.isna().sum() / 29692)

df = df.dropna()
"""

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("/Users/Will/Desktop/datathon/training.csv")


# Generate some test data

df2 = df[["surface_x","surface_y","OilPeakRate"]].dropna()

x = df2["surface_x"]
y = df2["surface_y"]

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.title("Density of Observations")
plt.show()

levels = np.linspace(df2["OilPeakRate"].min(), df2["OilPeakRate"].max(), 3)

plt.tripcolor(df2["surface_x"], df2["surface_y"], df2["OilPeakRate"])
plt.title("X Y Location vs Peak Flow")
plt.show()

########
#df = df.drop(["average_stage_length", "average_proppant_per_stage", "average_frac_fluid_per_stage", "number_of_stages","pad_id","ffs_frac_type","gross_perforated_length", "total_proppant", "total_fluid", "proppant_intensity", ],axis = 1)
df = df[df['OilPeakRate'].notna()]

df = df.set_index("pad_id")

print(df.isna().sum()/df.shape[0] * 100)

df = df.drop(["frac_seasoning", "average_stage_length", "average_proppant_per_stage", "average_frac_fluid_per_stage","number_of_stages","ffs_frac_type","Unnamed: 0","frac_type"],axis = 1)




df = df.dropna()

df2 = df.loc[df["frac_fluid_to_proppant_ratio"] != np.inf]


from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
# transform data
scaled1 = scaler1.fit_transform(df2[["surface_x","surface_y","bh_x","bh_y"]])

df3 = pd.DataFrame(scaled1, columns=df2[["surface_x","surface_y","bh_x","bh_y"]].columns, index = df2.index)

from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
# transform data
scaled2 = scaler2.fit_transform(df2.drop(["surface_x","surface_y","bh_x","bh_y","relative_well_position", "batch_frac_classification", "well_family_relationship","standardized_operator_name","OilPeakRate"],axis=1))

df4 = pd.DataFrame(scaled2, columns=df2.drop(["surface_x","surface_y","bh_x","bh_y","relative_well_position", "batch_frac_classification", "well_family_relationship","standardized_operator_name","OilPeakRate"],axis=1).columns, index = df2.index)

from sklearn.preprocessing import StandardScaler
scaler3 = StandardScaler()
# transform data
scaled3 = scaler3.fit_transform(df2[["OilPeakRate"]])

df5 = pd.DataFrame(scaled3, columns=["OilPeakRate"], index = df2.index)



df2 = pd.concat([df3,df4,df2[["relative_well_position", "batch_frac_classification", "well_family_relationship","standardized_operator_name"]],df5],axis=1)


dummy1 = pd.get_dummies(df2["relative_well_position"]).rename({"Unknown" : "Unknown1", "Standalone Well" : "StandaloneWell1"},axis=1)*1
df3 = pd.concat([df2,dummy1.set_index(df2.index)], axis=1)
dummy2 = pd.get_dummies(df2["batch_frac_classification"]).rename({"Unknown" : "Unknown2", "Standalone Well" : "StandaloneWell2"},axis=1)*1
df3 = pd.concat([df3,dummy2.set_index(df2.index)], axis=1)
dummy3 = pd.get_dummies(df2["well_family_relationship"]).rename({"Unknown" : "Unknown3", "Standalone Well" : "StandaloneWell3"},axis=1)*1
df3 = pd.concat([df3,dummy3.set_index(df2.index)], axis=1)
#df2 = pd.concat([df2,pd.get_dummies(df["standardized_operator_name"]).rename({"Unknown" : "Unknown4", "Standalone Well" : "StandaloneWell4"},axis=1)*1], axis=1)



df3 = df3.drop(["relative_well_position", "batch_frac_classification", "well_family_relationship","standardized_operator_name"],axis = 1)


#######
data_out = []
for i in range(100):
    
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df3, test_size=.20)


    from sklearn.neural_network import MLPRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split


    regr = MLPRegressor(max_iter=1000).fit(train.drop(["OilPeakRate"],axis = 1), train["OilPeakRate"])
    y_pred = regr.predict(test.drop(["OilPeakRate"],axis = 1))
    #print(regr.score(test.drop(["OilPeakRate"],axis = 1), test["OilPeakRate"]))

    from sklearn.metrics import mean_squared_error

    print(mean_squared_error(scaler3.inverse_transform(test[["OilPeakRate"]]), scaler3.inverse_transform(y_pred.reshape(-1, 1))) ** .5)
    data_out.append(mean_squared_error(scaler3.inverse_transform(test[["OilPeakRate"]]), scaler3.inverse_transform(y_pred.reshape(-1, 1))) ** .5)


result = test["OilPeakRate"] - y_pred

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# Clean data
df = pd.read_csv('training.csv')
scaler3 = StandardScaler()


def clean(df, scaler3):
    df = df[df['OilPeakRate'].notna()]

    df = df.set_index("pad_id")

    df = df.drop(["frac_seasoning", "average_stage_length", "average_proppant_per_stage", "average_frac_fluid_per_stage","number_of_stages","ffs_frac_type","Unnamed: 0","frac_type"],axis = 1)

    df = df.dropna()

    df2 = df.loc[df["frac_fluid_to_proppant_ratio"] != np.inf]

    scaler1 = MinMaxScaler()
    # transform data
    scaled1 = scaler1.fit_transform(df2[["surface_x","surface_y","bh_x","bh_y"]])

    df3 = pd.DataFrame(scaled1, columns=df2[["surface_x","surface_y","bh_x","bh_y"]].columns, index = df2.index)

    scaler2 = StandardScaler()
    # transform data
    scaled2 = scaler2.fit_transform(df2.drop(["surface_x","surface_y","bh_x","bh_y","relative_well_position", "batch_frac_classification", "well_family_relationship","standardized_operator_name","OilPeakRate"],axis=1))

    df4 = pd.DataFrame(scaled2, columns=df2.drop(["surface_x","surface_y","bh_x","bh_y","relative_well_position", "batch_frac_classification", "well_family_relationship","standardized_operator_name","OilPeakRate"],axis=1).columns, index = df2.index)

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
    # df2 = pd.concat([df2,pd.get_dummies(df["standardized_operator_name"]).rename({"Unknown" : "Unknown4",
    # "Standalone Well" : "StandaloneWell4"},axis=1)*1], axis=1)

    df3 = df3.drop(["relative_well_position", "batch_frac_classification", "well_family_relationship","standardized_operator_name"],axis = 1)

    return df3


# Split data
df3 = clean(df, scaler3)

n = len(df3)
train, test = train_test_split(df3, test_size=0.25, random_state=42)

# Random Forest Time :D
# numberTrees = 1000
# reg = RandomForestRegressor(n_estimators=numberTrees, random_state=42)
#
# reg.fit(train.drop(['OilPeakRate'], axis=1), train['OilPeakRate'])
#
# total_pred = reg.predict(test.drop(['OilPeakRate'], axis=1))
# best = root_mean_squared_log_error(test['OilPeakRate'].to_frame(), total_pred.reshape(-1, 1))
# print(best)

# Grid Search Time :D
# parameters = {'n_estimators': [200, 500],
#               'max_depth': [4, 8],
#               'max_features': ['auto', 'sqrt', 'log2']}
#
# grid = GridSearchCV(estimator=reg,
#                     param_grid=parameters,
#                     cv=3,
#                     scoring='neg_root_mean_squared_error')
#
# grid.fit(train.drop(['OilPeakRate'], axis=1), train['OilPeakRate'])
# print(grid.best_params_, grid.best_score_)

# Comparison between RF and Gradient Boost
# vals = {'Gradient Boosting': [],
#         'Random Forest': []}
# for i in range(50):
#     train, test = train_test_split(df3, test_size=0.25)
#
#     # # Make RF model
#     # rfr = RandomForestRegressor(n_estimators=800,
#     #                             n_jobs=-1)
#     # rfr.fit(train.drop(['OilPeakRate'], axis=1), train['OilPeakRate'])
#     #
#     # # Calculate RMSE for RF model
#     # rfr_pred = rfr.predict(test.drop(['OilPeakRate'], axis=1))
#     # vals['Random Forest'].append(root_mean_squared_error(scaler3.inverse_transform(test['OilPeakRate'].to_frame()),
#     #                                                      scaler3.inverse_transform(rfr_pred.reshape(-1, 1))))
#
#     # Make GB model
#     parameters = {"eta": 0.1,
#                   "max_depth": 9,
#                   "n_estimators": 300,
#                   "min_child_weight": 3}
#
#     gbr = xgb.XGBRegressor(n_estimators=800,
#                            parameters=parameters)
#     gbr.fit(train.drop(['OilPeakRate'], axis=1), train['OilPeakRate'])
#
#     # Calculate RMSE for GB model
#     gbr_pred = gbr.predict(test.drop(['OilPeakRate'], axis=1))
#     vals['Gradient Boosting'].append(root_mean_squared_error(scaler3.inverse_transform(test['OilPeakRate'].to_frame()),
#                                                              scaler3.inverse_transform(gbr_pred.reshape(-1, 1))))
#     print(root_mean_squared_error(scaler3.inverse_transform(test['OilPeakRate'].to_frame()),
#                                                              scaler3.inverse_transform(gbr_pred.reshape(-1, 1))))

# accuracy_df = pd.DataFrame(vals)
# accuracy_df.to_csv('comparison.csv')

df = pd.read_csv('scoring.csv')
df['OilPeakRate'] = 10000

scaler = StandardScaler()
scoring_df = clean(df, scaler)

n_estimators = 800

reg = RandomForestRegressor(n_estimators=n_estimators)
reg.fit(df3.drop(['OilPeakRate'], axis=1), df3['OilPeakRate'])

prediction = reg.predict(scoring_df.drop(['OilPeakRate'], axis=1))

pd.DataFrame({'OilPeakRate': list(scaler3.inverse_transform(prediction.reshape(-1, 1)))}).to_excel('submission_file.xlsx')

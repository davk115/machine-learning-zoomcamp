import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

import xgboost as xgb

# Importing Data
df_full = pd.read_csv('HR_data.csv')

df = df_full

# Cleaning Data
df.columns = df.columns.str.replace(' ', '_').str.lower()

df.rename(columns = {'satisfaction_level':'satisfaction_rating'}, inplace = True)
df.rename(columns = {'number_project':'number_of_projects'}, inplace = True)
df.rename(columns = {'average_montly_hours':'average_monthly_hours'}, inplace = True)
df.rename(columns = {'time_spend_company':'tenure'}, inplace = True)
df.rename(columns = {'work_accident':'work_accidents'}, inplace = True)
df.rename(columns = {'left':'termination'}, inplace = True)

# Training the model on the full dataset
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train_full = df_train_full.reset_index(drop=True)

y_train_full = df_train_full.termination.values
del df_train_full['termination']

dicts_train_full = df_train_full.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train_full = dv.fit_transform(dicts_train_full)

xgb_params = {
    'eta': 0.3,
    'max_depth': 5,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

dtrainfull = xgb.DMatrix(X_train_full, label=y_train_full,
                    feature_names = list(dv.get_feature_names_out()))

model = xgb.train(xgb_params, dtrainfull, num_boost_round=90,
                  verbose_eval=5,)

# Export the model

with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is exported to model.bin')
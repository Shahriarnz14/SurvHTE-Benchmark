'''
This script prepares the ACTG 175 dataset for causal survival forest analysis.
It fetches the dataset from the UCI Machine Learning Repository, processes it to add censoring,
and saves the processed data for three different treatment groups (HIV1, HIV2, and HIV3) into CSV files.
The script performs the following steps:
1. Fetch the ACTG 175 dataset from the UCI Machine Learning Repository.
2. Process the dataset to change the resolution from days to months.
3. Apply the effective non-censoring assumption to the dataset (it will result in the same effect as setting the `horizon` parameter in running CSF).
4. Add censoring to the dataset based on a Bernoulli distribution.
5. For each treatment group (HIV1, HIV2, and HIV3), run the Causal Survival Forest model 10 times,
    and take the average of the estimated conditional average treatment effect (CATE) as the base CATE.
6. Save the processed data for each treatment group into separate CSV files.
'''
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_causal_survival.causal_survival_forest import CausalSurvivalForestGRF


def insert_censoring(row, seed=2025):
    ### based on Mier et al. 2025
    # set random seed
    np.random.seed(int(0.4*seed) + row.name*5)
    # sample from bernoulli with p
    p = row['p']
    add_censor = np.random.binomial(1, p)
    if add_censor:
        # sample censoring time
        obs_time = row['observed_time_month']
        censor_time = np.random.uniform(1, min(obs_time, T_MAX/5))
        return censor_time, 0
    else:
        return row['observed_time_month'], row['effect_non_censor']
    
# fetch dataset https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
  
# data (as pandas dataframes) 
X = aids_clinical_trials_group_study_175.data.features 
y = aids_clinical_trials_group_study_175.data.targets 

# # metadata 
# print(aids_clinical_trials_group_study_175.metadata) 
  
# # variable information 
# print(aids_clinical_trials_group_study_175.variables) 


data = pd.concat([y, X], axis=1)

# change resolution to 1-month from 1-day (according to the Mier et al. 2025 paper)
data['observed_time_month'] = data['time']/30
# apply h=: effective non-censoring (Assumption 1 in CSF paper)
data['effect_non_censor'] = data['cid']
data.loc[data['observed_time_month'] >= 30, 'effect_non_censor'] = 1
data.loc[data['observed_time_month'] >= 30, 'observed_time_month'] = 30

T_MAX = data['observed_time_month'].max()


# prepare for adding censoring
data['p'] = 0.6 + 0.25 * data['z30']

# sample censoring time for 10 times
for seed in range(10):
    data[[f't{seed}', f'e{seed}']] = data.apply(insert_censoring, axis=1, result_type='expand', seed=seed)

data.drop(columns=['p'], inplace=True)

X_bi_cols = ['gender', 'race', 'hemo', 'homo', 'drugs', 'str2', 'symptom']
X_cont_cols = ['age', 'wtkg',  'karnof', 'cd40', 'cd80']
U_col = ['z30']
W_col = ['trt']
y_cols = ['observed_time_month', 'effect_non_censor'] # ['time', 'cid']
columns = y_cols + W_col + U_col + X_bi_cols + X_cont_cols + [f't{i}' for i in range(10)] + [f'e{i}' for i in range(10)]
failure_times_grid_size = 200

## HIV 1: ZDV(0) vs ZDV + ddI(1)
hiv1 = data[data.trt.isin([0,1])][columns].copy()
# assign numbered idx to each row
hiv1['id'] = range(1, len(hiv1) + 1)
hiv1 = hiv1[['id'] + columns]
hiv1['trt'] = hiv1['trt'].apply(lambda x: 0 if x == 0 else 1)
X = hiv1[X_bi_cols + X_cont_cols].values
W = hiv1[W_col].values.flatten()
Y = hiv1[y_cols].values
cate_hat = np.zeros((X.shape[0], 10))
# run csf for 10 times, take the average cate --> base cate
for i in range(10):
    csf = CausalSurvivalForestGRF(failure_times_grid_size=failure_times_grid_size, horizon=30, min_node_size=18, seed=i)
    csf.fit(X, W, Y)
    cate_hat[:, i] = csf.predict_cate(X)
cate_base = cate_hat.mean(axis=1) # used as true cate for comparison with est. cate after adding censoring
hiv1['cate_base'] = cate_base
hiv1.to_csv('ACTG_175_HIV1.csv', index=False)

## HIV 2: ZDV(0) vs ZDV + Zal(2)
hiv2 = data[data.trt.isin([0,2])][columns].copy()
hiv2['id'] = range(1, len(hiv2) + 1)
hiv2 = hiv2[['id'] + columns]
hiv2['trt'] = hiv2['trt'].apply(lambda x: 0 if x == 0 else 1)
X = hiv2[X_bi_cols + X_cont_cols].values
W = hiv2[W_col].values.flatten()
Y = hiv2[y_cols].values
cate_hat = np.zeros((X.shape[0], 10))
# run csf for 10 times, take the average cate --> base cate
for i in range(10):
    csf = CausalSurvivalForestGRF(failure_times_grid_size=failure_times_grid_size, horizon=30, min_node_size=18, seed=i)
    csf.fit(X, W, Y)
    cate_hat[:, i] = csf.predict_cate(X)
cate_base = cate_hat.mean(axis=1) # used as true cate for comparison with est. cate after adding censoring
hiv2['cate_base'] = cate_base
hiv2.to_csv('ACTG_175_HIV2.csv', index=False)

## HIV 2: ZDV(0) vs ddI(3)
hiv3 = data[data.trt.isin([0,3])][columns].copy()
hiv3['id'] = range(1, len(hiv3) + 1)
hiv3 = hiv3[['id'] + columns]
hiv3['trt'] = hiv3['trt'].apply(lambda x: 0 if x == 0 else 1)
X = hiv3[X_bi_cols + X_cont_cols].values
W = hiv3[W_col].values.flatten()
Y = hiv3[y_cols].values
cate_hat = np.zeros((X.shape[0], 10))
# run csf for 10 times, take the average cate --> base cate
for i in range(10):
    csf = CausalSurvivalForestGRF(failure_times_grid_size=failure_times_grid_size, horizon=30, min_node_size=18, seed=i)
    csf.fit(X, W, Y)
    cate_hat[:, i] = csf.predict_cate(X)
cate_base = cate_hat.mean(axis=1) # used as true cate for comparison with est. cate after adding censoring
hiv3['cate_base'] = cate_base
hiv3.to_csv('ACTG_175_HIV3.csv', index=False)

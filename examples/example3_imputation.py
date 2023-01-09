"""
Imputation of missing values for data/diabetes_amputed.csv
"""

import pandas as pd
from src import mice

df_amp = pd.read_csv("data/diabetes_amputed.csv")
X_amp = df_amp.to_numpy()

m_imputations = 10
n_iterations = 20
imp = mice.mice(X_amp, n_iterations, m_imputations, 42)

columns = df_amp.columns

df_all_imputations = []

for m in range(m_imputations):
    df_single_imputation = pd.DataFrame(data=imp[m], columns=columns)
    df_single_imputation = df_single_imputation.assign(imputation=m)
    df_all_imputations.append(df_single_imputation)

df_all_imputations = pd.concat(df_all_imputations, ignore_index=True)

df_all_imputations.to_csv("data/diabetes_imputed.csv", index=False)

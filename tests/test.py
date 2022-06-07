
def test_sample_data():
  import pandas as pd
  import numpy as np
  from src import mice
  
  df = pd.read_csv("/home/runner/work/mice/mice/tests/data.csv")
  df_amp = pd.read_csv("/home/runner/work/mice/mice/tests/data_amputed.csv")
  
  df_np = df.to_numpy()
  X_amp = df_amp.to_numpy()

  m_imputations = 10
  n_iterations = 20
  imp = mice.mice(X_amp, n_iterations, m_imputations, 42)
  
  missing_map = np.isnan(X_amp)
  targets = df["hn4_qol"][missing_map[:, 3]].to_numpy()

  for m in range(m_imputations):
    predictions = imp[m][missing_map[:, 3], 3]

    print("RMSE for hn4_qol variable:", np.sqrt(np.mean((predictions-targets)**2)))

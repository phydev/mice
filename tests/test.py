
def test_sample_data():
  import pandas as pd
  import numpy as np
  from pyampute.ampute import MultivariateAmputation
  from src import mice
  
  df = pd.read_csv("/home/runner/work/mice_from_scratch/mice_from_scratch/tests/data.csv")
  df_np = df.to_numpy()
  
  ma = MultivariateAmputation(seed=42)
  X_amp = ma.fit_transform(df_np)

  df_amp = pd.DataFrame(X_amp, columns=df.columns, dtype=np.float64)

  m_imputations = 10
  n_iterations = 20
  imp = mice.mice(X_amp, n_iterations, m_imputations, 42)
  
  missing_map = np.isnan(X_amp)
  targets = df["hn4_qol"][missing_map[:, 3]].to_numpy()

  for m in range(m_imputations):
    predictions = imp[m][missing_map[:, 3], 3]

    print("RMSE for hn4_qol variable:", np.sqrt(np.mean((predictions-targets)**2)))

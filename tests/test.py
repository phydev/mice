
def test_sample_data():
  import pandas as pd
  from pyampute.ampute import MultivariateAmputation
  from src import mice
  
  df = pd.read_csv("/home/runner/work/mice_from_scratch/mice_from_scratch/tests/data.csv")
  df_np = df.to_numpy()
  
  ma = MultivariateAmputation(seed=42)
  X_amp = ma.fit_transform(df_np)

  df_amp = pd.DataFrame(X_amp, columns=df.columns, dtype=np.float64)

  imp = mice.mice(X_amp, 20, 10, 42)

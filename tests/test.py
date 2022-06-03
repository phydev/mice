from pyampute.ampute import MultivariateAmputation
from src import *

ma = MultivariateAmputation(seed=42)
X_amp = ma.fit_transform(df_np)

df_amp = pd.DataFrame(X_amp, columns=df.columns, dtype=np.float64)

imp = mice(X_amp, 20, 10, 42)

from sklearn.datasets import load_iris
from pyampute.ampute import MultivariateAmputation
from src import mice

iris = load_iris(as_frame=True, return_X_y=False)["data"]
ma = MultivariateAmputation()
X_amp = ma.fit_transform(iris.to_numpy()) # pyampute requires the input as numpy array

imp = mice.mice(X, n_iterations = 20, m_imputations = 10, seed=42)
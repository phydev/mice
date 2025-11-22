[![Build and Deploy](https://github.com/phydev/mice/actions/workflows/python-app.yml/badge.svg)](https://github.com/phydev/mice/actions/workflows/python-app.yml)
# MICE - Multiple Imputation by Chained Equations
Multiple imputation by chained equation implemented from scratch. 

## Example 1: iris dataset

Load the iris data from sklearn and introduce missing values with [pyampute package](https://github.com/RianneSchouten/pyampute)
```python
from sklearn.datasets import load_iris
from pyampute.ampute import MultivariateAmputation

iris = load_iris(as_frame=True, return_X_y=False)["data"]
ma = MultivariateAmputation()
X_amp = ma.fit_transform(iris.to_numpy()) # pyampute requires the input as numpy array

```
Now we can apply MICE in the amputed dataset
```python
from src import mice
imp = mice.mice(X, n_iterations = 20, m_imputations = 10, seed=42)
```


## Example 2: distribution plot for the sample data
After imputation you should make diagnostic plots and check the distribution of the multiply imputed datasets comparing with the complete case data. Bellow you can find the plot for the example we provide in  /tests directory:

```python
import seaborn as sns
import matplotlib.pyplot as plt

p = 3 # column to be plotted
custom_lines = [plt.Line2D([0], [0], color="red", lw=4),
                plt.Line2D([0], [0], color="grey", lw=4),
                plt.Line2D([0], [0], color="blue", lw=4)]

fig, ax = plt.subplots()

for m in range(len(imp)):
    sns.kdeplot(imp[m][:, p], label="Imputed", color="black", lw=0.2, ax=ax)
sns.kdeplot(X_amp[:,p], label="Missing", color="blue", ax=ax)
sns.kdeplot(df.to_numpy()[:, p], label="Complete", color="red",ax=ax)
plt.xlabel("Age (years)")
ax.legend(custom_lines, ['Complete', 'Imputed', 'Missing'], loc="upper left")
plt.savefig("qol_distribution_mice.png")
```

![Figure showing the distribution lines for 10 imputed datasets, the original dataset and the amputed dataset with missing values.](https://github.com/phydev/mice/blob/main/tests/qol_distribution_mice.png)

## Beware
This is a low performance implementation meant for pedagogical purposes only. There are several limitations and improvements that can be made, for research please use one of the available packages for multiple imputation:
- [mice](https://cran.r-project.org/web/packages/mice/index.html)
- [miceRanger](https://github.com/FarrellDay/miceRanger)
- [sklearn.imputer](https://scikit-learn.org/stable/modules/impute.html)



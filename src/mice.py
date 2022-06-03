import numpy as np
import matplotlib.pyplot as plt

def compute_ecdf(x):
    """
    computes the empirical cumulative density function for x
    :param x: column of observations
    :return A: matrix with the empirical cumulative density function
    """
    xc = np.sort(x)
    yc = np.arange(1, len(xc)+1)/len(xc)
    
    A = np.array([xc, yc]).T
    
    return A

def gibbs_sampler(x, n_samples, plot=False):
    """
    compute the empirical cumulative density function for the distribution x 
    and draw n_samples
    """
    
    if n_samples<1:
        return 0

    empirical_cdf = compute_ecdf(x)
    sample = np.zeros(n_samples)
    n_sample = 0
    
    if plot:
        plt.plot(empirical_cdf.T[0], empirical_cdf.T[1])
        plt.title("Empirical Cumulative Density Function")
        plt.ylabel("ECDF")
        plt.xlabel("Observable")
        plt.show()

    for i in range(0, n_samples):
        random_number = np.random.rand()
       
        while np.isclose(np.round(random_number, decimals=2), 0.00, rtol=10e-2):
            random_number = np.random.rand()
            
        for pair in empirical_cdf:
            if(np.isclose(np.around(pair[1], decimals=2), np.around(random_number, decimals=2), rtol=10e-3)):
                sample[n_sample] = pair[0]
        
                n_sample += 1
                
                break  
    if plot:
        sns.histplot(a, bins="auto", label="Sample", color="red")
        sns.histplot(df["hn1_age"], bins="auto", label="Data")
        plt.title("Gibbs sample")
        plt.legend()
        plt.show()
    return sample


def replace_nans(X, missing_map, func = np.nanmean, **kwargs):
    """
    replace missing values by the mean or other quantity, e.g. np.nanmedian
    """
    for col in range(X.shape[1]):
        mean = func(X[:, col], **kwargs)
        for row in range(X.shape[0]):
            if(missing_map[row, col]):
                X[row, col] = mean 
            
    return X

def fit_covariate_imputation(X, missing_map):
    """
    fit imputation model for each variable
    """
    
    from sklearn import tree
    X_imp = np.copy(X)
    X = replace_nans(X, missing_map)
    
    for col in range(X.shape[1]):
        
        X_subset = X[~missing_map[:, col]]
        
        y = X_subset[:, col] # selecting available outcomes for fitting
        X_fit = np.delete(X_subset, col, axis=1)
            
        reg = tree.DecisionTreeRegressor()
        reg = reg.fit(X_fit, y.reshape(-1,1))
        
        X_fit_all = np.delete(X, col, axis=1)
        
        for row in range(X.shape[0]):
            
            if(missing_map[row, col]):
                X_imp[row, col] = reg.predict(X_fit_all[row,:].reshape(1, -1))
    
    return X_imp


def perform_iterations(X, n_iterations=5):
    """
    Perform n iterations of chained imputation 
    """
    
    missing_map = np.isnan(X)
    X_imp = np.copy(X)
    hn4_qol = np.zeros(n_iterations)
    for iteration in range(n_iterations):

        X_imp = fit_covariate_imputation(X_imp, missing_map)
        

        for col in range(X_imp.shape[1]):
            n_samples = np.count_nonzero(missing_map[:, col])
            samples = gibbs_sampler(X_imp[:, col], n_samples, plot=False)
            n = 0
            for row in range(X_imp.shape[0]):
                if(missing_map[row, col]):
                    X_imp[row, col] = samples[n]
                    n+=1
            
        
        hn4_qol[iteration] = X_imp[2,3]
    
    plt.plot(hn4_qol)
    
    return X_imp
    
def mice(X, n_iterations, m_imputations, seed):
    """
    perform m imputations with chained equations
    """
    np.random.seed(seed)
    
    imp = []
    for m in range(m_imputations):
        print("imputation:", m)
        X_imp = perform_iterations(X, n_iterations)
        imp.append(X_imp)
        
    return imp

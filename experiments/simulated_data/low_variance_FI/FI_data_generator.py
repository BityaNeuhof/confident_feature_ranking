import numpy as np
import pandas as pd
from experiments.simulated_data.data_generator import gen_data_ranking, gen_cov_matrix

## Generate data for feature importance simulations
def gen_data_X_unif(n, p, seed, a=0, b=1):
    np.random.seed(seed)
    if (((type(a) is int) or (type(a) is float)) and ((type(b) is int) or (type(b) is float))):
        X = np.random.uniform(low=float(a), high=float(b), size=(n, p))
        X = pd.DataFrame(X, columns=['X' + str(i + 1) for i in range(p)])
    else:
        if ((len(a) != len(b)) or (len(a) != p)):
            raise ValueError('Error: len(a) and len(b) (%d) must be the same as p (%d)', len(a), len(b), p)
        X = pd.DataFrame()
        for i in range(p):
            np.random.seed(seed * (i + 1))
            X['X' + str(i + 1)] = np.random.uniform(low=float(a[i]), high=float(b[i]), size=n)
    return X


def gen_data_X_normal(n, p, seed, mus=None, cov_matrix=None):
    if mus is None:
        mus = np.zeros(p)
    if cov_matrix is None:
        cov_matrix = np.identity(p)
    # The seed is used for sampling random data.
    X = gen_data_ranking(mus, cov_matrix, n, seed)
    feature_names = ['X' + str(i + 1) for i in range(p)]
    X.columns = feature_names
    return X


def gen_X1(n, p, seed):
    return gen_data_X_unif(n, p, seed)

def f1(X, seed):
    n, p = X.shape
    y = np.zeros(n)
    cycle_size = 10
    max_imp_features = p - p % cycle_size
    for i in range(0, max_imp_features, cycle_size):
        y += (10 * np.sin(np.pi * X[:,i] * X[:,i+1])
              + 20 * np.power(X[:,i+2] - 0.5, 2)
              + 10 * X[:,i+3]
              + 5 * X[:,i+4])
    # add noise
    np.random.seed(seed)
    y += np.random.standard_normal(n)
    return y


def gen_X2(n, p, seed):
    a = [0, np.pi, 0, 1, 0, 0, 0, 0, 0, 0]
    b = [10, 2 * np.pi, 1, 5, 1, 1, 1, 1, 1, 1]
    cycle_size = 10

    a = np.tile(a, p // cycle_size)
    b = np.tile(b, p // cycle_size)

    # noise variables
    a = np.concatenate([a, np.zeros(p % cycle_size)])
    b = np.concatenate([b, np.ones(p % cycle_size)])

    return gen_data_X_unif(n, p, seed, a, b)

def f2(X, seed):
    n, p = X.shape
    y = np.zeros(n)
    cycle_size = 10
    max_imp_features = p - p %  cycle_size
    for i in range(0, max_imp_features,  cycle_size):
        y += (np.sqrt(np.power(X[:,i], 2) 
                      + np.power((X[:,i+1] * X[:,i+2] - 1 / (X[:,i+1] * X[:,i+3])), 2)))

    # add noise
    np.random.seed(seed)
    y += np.random.standard_normal(n)
    return y


def gen_X3(n, p, seed):
    return gen_data_X_normal(n, p, seed, cov_matrix=gen_cov_matrix(p, rho=0.3, 
                                                                   corr_type='off_diag', sigma_type='chi', seed=seed))

def f3(X, seed):
    n, p = X.shape
    y = np.zeros(n)
    cycle_size = 10
    max_imp_features = p - p %  cycle_size
    for i in range(0, max_imp_features,  cycle_size):
        y += (X[:,i] * X[:,i+1]
              + np.power(X[:,i+2], 2)
              - X[:,i+3] * X[:,i+6]
              + X[:,i+7] * X[:,i+9]
              - np.power(X[:,i+5], 2))
    # add noise
    np.random.seed(seed)
    y += np.random.standard_normal(n)
    return y


def gen_data_fi(n, p, func_ind, seed):
    X_func = eval('gen_X' + str(func_ind))
    y_func = eval('f' + str(func_ind))
    data = X_func(n, p, seed)
    y = y_func(data.values, seed)
    data['target'] = y
    return data, 'target'




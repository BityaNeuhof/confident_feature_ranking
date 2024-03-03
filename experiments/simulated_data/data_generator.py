import numpy as np
import pandas as pd

## Generate data for ranking methods simulations
def gen_data_ranking(mus, cov_matrix, n, seed):
    np.random.seed(seed)
    return pd.DataFrame(np.random.multivariate_normal(mean=mus, cov=cov_matrix, size=n))


def gen_mu_vec(p, seed, exponent=0.5, ties=False, zeros=False):
    mus = np.power(np.array(range(1, p + 1)), exponent)
    if zeros: # simulate zeros
        actual_rank = np.ceil(p / 2).astype(int)
        mus = np.concatenate([mus[:actual_rank + 1], np.zeros(p - actual_rank)])
    if ties: # simulate ties
        np.random.seed(seed)
        mus = np.random.choice(mus, p)
    return mus


def gen_cov_matrix(p, rho, corr_type='id', sigma_type='ones', factor=1, seed=0):
    if corr_type not in ['id', 'pairs', 'off_diag']:
        raise NameError('corr_type %s is not defined.', corr_type)
    if sigma_type not in ['ones', 'unif', 'chi']:
        raise NameError('sigma_type %s is not defined.', sigma_type)
    corr_matrix = np.eye(p)
    if corr_type == 'pairs':
        for i, j in zip(range(0, p - 1, 2), range(1, p, 2)):
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
    elif corr_type == 'off_diag':
        corr_matrix[np.where(corr_matrix == 0)] = rho
    sigmas = np.ones(p)
    if sigma_type == 'unif':
        np.random.seed(seed)
        sigmas = np.random.uniform(0.2, 1.0, p)
    elif sigma_type == 'chi':
        np.random.seed(seed)
        sigmas = np.sqrt(np.random.chisquare(5, p) / 5)

    sigmas = factor * sigmas
    cov_matrix = np.diag(sigmas) @ corr_matrix @ np.diag(sigmas)
    cov_matrix = np.round(cov_matrix, 4)
    if not ((np.array_equal(cov_matrix, cov_matrix.T)) and (np.all(np.linalg.eigvals(cov_matrix) >= 0))):
        print(p, rho, corr_type, sigma_type, factor, seed, 'covariance is not positive semi-definite.')
        return None
    return cov_matrix
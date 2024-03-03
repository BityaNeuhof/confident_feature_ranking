import copy
from itertools import product
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from utils.icranks import calc_icranks
from utils.feature_ranking import confident_simultaneous_ranking, bootstrap_ranking
from utils.general import get_ranks_measures
from experiments.simulated_data.data_generator import gen_cov_matrix, gen_mu_vec, gen_data_ranking


def ranking_comparison_simulation_iteration(p, rho, corr_type, sigma_type, factor, n, ties,
                                            methods, alpha, mu_exponent, rep_seed, rep_bootstrap, seed):
    cov_matrix = gen_cov_matrix(p, rho, corr_type, sigma_type, factor, seed)
    # skip configuration if the covariance matrix is not positive semi-definite
    if cov_matrix is None:
        return []
    mus = gen_mu_vec(p, seed, mu_exponent, ties)

    ranks_measures_res = []
    iter_params = {'p': p, 'n': n,
                   'rho': rho, 'corr_type': corr_type,
                   'sigma_type': sigma_type, 'factor': factor,
                   'ties': ties, 'mu_exponent': mu_exponent,
                   'rep_bootstrap': rep_bootstrap, 'rep_seed': rep_seed}
    data = gen_data_ranking(mus, cov_matrix, n, rep_seed)
    for method in methods:
        res = None
        if method == 'icranks':
            n = data.shape[0]
            means = data.mean(axis=0).values
            means_ord = np.argsort(means)
            stds = (np.std(data, axis=0, ddof=1) / np.sqrt(n)).values
            res = calc_icranks(means, stds, means_ord, 'Tukey', alpha=alpha)
        elif method == 'bootstrap':
            res = bootstrap_ranking(data.to_numpy(), alpha, rep_bootstrap)
        else:
            res = confident_simultaneous_ranking(data.to_numpy(), alpha, method=method, rep_bootstrap=rep_bootstrap)
        ranks_measures_res.append({**iter_params, **{'rank_method': method},
                                   **get_ranks_measures(mus, res)})
    return ranks_measures_res


def run_ranking_comparison_simulation(params):
    # Create all combinations
    all_configurations = product(*list(params.values()))
    config_keys = params.keys()

    ranks_measures = []
    def add_result(result):
        ranks_measures.extend(result)
        if len(ranks_measures) % 100 == 0:
            print(len(ranks_measures))

    def get_seed(config_dict):
        seed = config_dict['p']
        return seed

    with Pool(processes=cpu_count()) as pool:
        for config in all_configurations:
            full_config = dict(zip(config_keys, config))
            full_config['seed'] = get_seed(full_config)
            rep = full_config.pop('rep')
            for i in range(rep):
                iter_config = copy.deepcopy(full_config)
                iter_config['rep_seed'] = i + 1
                pool.apply_async(ranking_comparison_simulation_iteration,
                                 kwds=iter_config, callback=add_result)

        pool.close()
        pool.join()
    return pd.DataFrame(ranks_measures)
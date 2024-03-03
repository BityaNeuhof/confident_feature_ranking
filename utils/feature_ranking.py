from itertools import permutations, combinations
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
from scipy.stats import rankdata


def calc_t_test(data, alternative='less'):
    p = data.shape[1]
    t_res = {}
    for pair in permutations(range(1 , p + 1), 2):
        pair_diff = data[:, pair[0] - 1] - data[:, pair[1] - 1]
        if np.min(pair_diff) == np.max(pair_diff):
            t_res[pair] = {'statistic' : 0, 'pvalue': 1}
        else:
            statistic, pvalue = ttest_rel(data[:, pair[0] - 1], data[:, pair[1] - 1], alternative=alternative)
            t_res[pair] = {'statistic' : statistic, 'pvalue': pvalue}
    return pd.DataFrame(t_res).T.reset_index().rename(columns={'level_0' : 'var_1', 'level_1': 'var_2'})


def min_p_correction(paired_test_res, alpha, rep_bootstrap, base_values,
                     paired_test_func=calc_t_test, alternative='less'):
    orig_ps = paired_test_res['pvalue']
    paired_test_res['adj_pvalue'] = np.nan
    paired_test_res['reject'] = False
    orig_ps_order = np.argsort(orig_ps)
    bootstraped_ps = []

    base_values_null = base_values - np.mean(base_values, axis=0)
    for i in range(rep_bootstrap):
        base_values_null_i = pd.DataFrame(base_values_null).sample(frac=1, replace=True).values
        bootstraped_ps.append(paired_test_func(base_values_null_i, alternative=alternative)['pvalue'])

    bootstraped_ps = np.array(bootstraped_ps)
    prev_adj_pvalue = 0
    for p_ind in orig_ps_order:
        min_p = orig_ps.iloc[p_ind]
        min_bootstraped_ps = np.min(bootstraped_ps, axis=1)
        min_p_lower = min_bootstraped_ps <= min_p
        adj_pvalue = np.sum(min_p_lower) / rep_bootstrap
        adj_pvalue = max(adj_pvalue, prev_adj_pvalue) # keep the sequence monotonically increasing
        paired_test_res.at[p_ind, 'adj_pvalue'] = adj_pvalue
        paired_test_res.at[p_ind, 'reject'] = adj_pvalue <= alpha
        bootstraped_ps[:,p_ind] = 1
        prev_adj_pvalue = adj_pvalue

    return paired_test_res


def multi_hypothesis_testing(paired_test_res, alpha, method='holm'):
    multi_res = multipletests(paired_test_res['pvalue'], alpha, method=method)
    paired_test_res['adj_pvalue'] = multi_res[1]
    paired_test_res['reject'] = multi_res[0]
    return paired_test_res


def calc_set_ranks(p, multi_test_res):
    """
    If the pair (j, k) in multi_test_res with reject=True, then imp_j < imp_k.
    """
    lowers = np.full(p, 1) # 1 for all
    uppers = np.full(p, p) # p for all
    for pair in combinations(range(1, p + 1), 2):
        j = pair[0]
        k = pair[1]
        test_res_jk = multi_test_res[(multi_test_res['var_1'] == j) & (multi_test_res['var_2'] == k)]['reject'].values[0]
        test_res_kj = multi_test_res[(multi_test_res['var_1'] == k) & (multi_test_res['var_2'] == j)]['reject'].values[0]
        if not (test_res_jk or test_res_kj): # j = k
            continue
        elif test_res_jk: # j < k
            uppers[j - 1] -= 1
            lowers[k - 1] += 1
        else: # k < j
            uppers[k - 1] -= 1
            lowers[j - 1] += 1
    return pd.DataFrame([lowers, uppers]).T.rename(columns={0: 'L', 1:'U'})


def confident_simultaneous_ranking(base_values, alpha, method='holm',
                                   paired_test_func=calc_t_test, alternative='less',
                                   rep_bootstrap=None):
    p = base_values.shape[1] # number of features
    paired_test_res = paired_test_func(base_values, alternative=alternative)
    # print(paired_test_res)
    multi_res = None
    if method == 'min_p':
        multi_res = min_p_correction(paired_test_res.copy(), alpha, rep_bootstrap,
                                     base_values, paired_test_func, alternative)
    else:
        multi_res = multi_hypothesis_testing(paired_test_res.copy(), alpha, method)
    # multi_res.to_csv('results_paper/simulated_data/p_' + str(p) + '/raw/multi_res.csv')
    set_ranks = calc_set_ranks(p, multi_res)
    return set_ranks


def bootstrap_ranking(base_values, alpha, rep):
    all_ranks = []
    for i in range(rep):
        np.random.seed(i + 1)
        inds = np.random.choice(base_values.shape[0], base_values.shape[0], replace=True)
        base_values_i = base_values[inds,:]
        means_i = np.mean(np.abs(base_values_i), axis=0)
        all_ranks.append((rankdata(means_i)).astype(int))
    all_ranks = pd.DataFrame(all_ranks)
    lowers = all_ranks.quantile(alpha / 2, axis=0).values.astype(int)
    uppers = all_ranks.quantile(1 - (alpha / 2), axis=0).values.astype(int)
    return pd.DataFrame([lowers, uppers]).T.rename(columns={0: 'L', 1: 'U'})


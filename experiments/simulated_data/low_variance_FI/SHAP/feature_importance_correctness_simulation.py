import pickle, copy, time
from itertools import product
from multiprocessing import cpu_count, Pool
import pandas as pd
import numpy as np
from utils.feature_ranking import confident_simultaneous_ranking
from utils.feature_importance import get_shap_global
from utils.icranks import calc_icranks
from utils.general import get_ranks_measures
from experiments.simulated_data.low_variance_FI.FI_data_generator import gen_data_fi


def ranking_fi_simulation_iteration(n, p, func_ind, model_type, methods, alpha, 
                                    rep_seed, rep_bootstrap, 
                                    model_config_path, true_importance_path):
    model_params = {'model_type': model_type, 'func_ind': func_ind, 'p': p}
    model_params_str = '_'.join([str(val) for val in model_params.values()])
    model_config_filename = model_config_path + model_params_str + '.sav'
    model_config = pickle.load(open(model_config_filename, 'rb'))

    # Load model and explainer
    explainer = model_config['explainer']

    # 'True' importance
    true_filename = true_importance_path + 'true_feature_importance_all_models.npy'
    true_importance = np.load(true_filename, allow_pickle=True).item()
    shap_true = true_importance[model_params_str]['shap_means_true']

    ranks_measures_shap = []
    iter_params = {'p': p, 'n': n, 'model_type': model_type, 'func_ind': func_ind, 
                   'alpha': alpha, 'rep_bootstrap': rep_bootstrap, 'rep_seed': rep_seed}
    data, target_name = gen_data_fi(n, p, func_ind, seed=rep_seed)
    feature_names = data.columns.drop(target_name)
    times = {}
    # TreeSHAP
    st = time.process_time()
    _, _, shap_values = get_shap_global(explainer, data[feature_names], return_local=True)
    base_values = np.abs(shap_values)
    times['shap_time'] = time.process_time() - st
    for method in methods:
        if method == 'icranks':
            st = time.process_time()
            n = base_values.shape[0]
            means = np.mean(base_values, axis=0)
            means_ord = np.argsort(means)
            stds = (np.std(base_values, axis=0, ddof=1) / np.sqrt(n))
            np.place(stds, stds == 0, np.finfo(float).eps)
            shap_set_ranks = calc_icranks(means, stds, means_ord, 'Tukey', alpha=alpha)
            times['icranks_time'] = time.process_time() - st
        
        else:
            st = time.process_time()
            shap_set_ranks = confident_simultaneous_ranking(base_values, alpha, method=method,
                                                            rep_bootstrap=rep_bootstrap)
            times[method + '_time'] = time.process_time() - st
        ranks_measures_shap.append({**iter_params, **{'rank_method': method}, 
                                    **{'importance_method': 'shap'},
                                    **get_ranks_measures(shap_true, shap_set_ranks), **times})
    return ranks_measures_shap


def run_ranking_fi_simulation(params):
    # Create all combinations
    all_configurations = product(*list(params.values()))
    config_keys = params.keys()

    ranks_measures = []
    def add_result(result):
        ranks_measures.extend(result)
        if len(ranks_measures) % 50 == 0:
            print(len(ranks_measures))


    with Pool(processes=cpu_count()) as pool:
        for config in all_configurations:
            config = dict(zip(config_keys, config))
            rep = config.pop('rep')
            for i in range(rep):
                iter_config = copy.deepcopy(config)
                iter_config['rep_seed'] = i + 1
                pool.apply_async(ranking_fi_simulation_iteration,
                                 kwds=iter_config, callback=add_result)

        pool.close()
        pool.join()
    return pd.DataFrame(ranks_measures)

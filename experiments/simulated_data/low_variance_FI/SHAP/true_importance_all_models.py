import pickle
from itertools import product
from multiprocessing import cpu_count, Pool
from utils.feature_importance import get_shap_global
from experiments.simulated_data.low_variance_FI.FI_data_generator import gen_data_fi



def calc_true_importance(n_true, p, func_ind, model_type, model_config_path, seed):
    model_params = {'model_type': model_type, 'func_ind': func_ind, 'p': p}
    model_params_str = '_'.join([str(val) for val in model_params.values()])
    model_config_filename = model_config_path + model_params_str + '.sav'

    model_config = pickle.load(open(model_config_filename, 'rb'))

    # Load explainer
    explainer = model_config['explainer']

    true_res = {}
    # 'True' importance
    data_true, target_name = gen_data_fi(n_true, p, func_ind, seed)
    feature_names = data_true.columns.drop(target_name)

    # TreeSHAP
    shap_means_true, _ = get_shap_global(explainer, data_true[feature_names], return_local=False)
    true_res['shap_means_true'] = shap_means_true
    return {model_params_str: true_res}


def true_importance_all_models(params):
    # Create all combinations
    all_configurations = product(*list(params.values()))
    config_keys = params.keys()

    true_importance = {}
    def add_result(result):
        print(result.keys())
        true_importance.update(result)

    def get_seed(config_dict):
        seed = config_dict['p']
        return seed

    with Pool(processes=cpu_count()) as pool:
        for config in all_configurations:
            full_config = dict(zip(config_keys, config))
            full_config['seed'] = get_seed(full_config)
            pool.apply_async(calc_true_importance,
                             kwds=full_config, callback=add_result)
        pool.close()
        pool.join()
    return true_importance

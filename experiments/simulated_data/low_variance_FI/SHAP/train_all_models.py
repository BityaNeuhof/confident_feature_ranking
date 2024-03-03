import pickle, os
from itertools import product
from multiprocessing import cpu_count, Pool
import pandas as pd
import numpy as np
import shap
from utils.general import train_model
from experiments.simulated_data.low_variance_FI.FI_data_generator import gen_data_fi


def train_save_model(n_learning, p, func_ind, model_type, model_config_path):
    model_params = {'model_type': model_type, 'func_ind': func_ind, 'p': p}
    model_params_str = '_'.join([str(val) for val in model_params.values()])
    os.makedirs(model_config_path, exist_ok=True)
    model_config_filename = model_config_path + model_params_str + '.sav'

    model_config = {}

    np.random.seed(p)
    learning_data, target_name = gen_data_fi(n_learning, p, func_ind, seed=2023)
    feature_names = learning_data.columns.drop(target_name)
    # Fit an model
    model, model_scores = train_model(learning_data, feature_names, 
                                      target_name, model_type, {'random_state': 2023})
    model_config['model'] = model
    # build an explainer (local TreeExplainer)
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    model_config['explainer'] = explainer
    pickle.dump(model_config, open(model_config_filename, 'wb'))
    model_params.update(model_scores)
    return model_params


def train_all_models(params):
    all_configurations = product(*list(params.values()))
    config_keys = params.keys()

    models_scores = []
    def add_result(result):
        models_scores.append(result)

    with Pool(processes=cpu_count()) as pool:
        for config in all_configurations:
            pool.apply_async(train_save_model, kwds=dict(zip(config_keys, config)), callback=add_result)
        pool.close()
        pool.join()
    return pd.DataFrame(models_scores)
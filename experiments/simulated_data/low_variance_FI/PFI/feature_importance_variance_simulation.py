from itertools import product
import numpy as np
import pandas as pd
from utils.feature_importance import permute_all_features_obs
from experiments.simulated_data.low_variance_FI.FI_data_generator import gen_data_fi, f1, f2
from experiments.simulated_data.low_variance_FI.PFI.dummy_model import DummyRegressor


def run_pfi_base_to_global_variance_simulation(ns, p, func_inds, Bs, rep, train_seed, is_reg = True):
    all_res = []
    for func_ind in func_inds:
        model = DummyRegressor(eval('f' + str(func_ind)))
        train_data, target_name = gen_data_fi(ns[0], p, func_ind, train_seed)
        feature_names = train_data.columns.drop(target_name)
        X  = train_data[feature_names]
        y = train_data[target_name].values
        model.fit(X, y)
        
        for (n, B) in product(ns, Bs):
            global_scores = {feature: [] for feature in feature_names}
            for r in range(rep):
                seed = r + 1
                data, target_name = gen_data_fi(n, p, func_ind, seed)
                X  = data[feature_names]
                y = data[target_name].values
                all_perms_by_feature = permute_all_features_obs(model, X, y, B, is_reg, seed, return_perms=True)
                for feature in all_perms_by_feature.keys():
                    base_values = np.mean(all_perms_by_feature[feature], axis=0) # mean over permutations
                    global_score = np.mean(base_values)
                    global_scores[feature].append(global_score)
            global_scores = pd.DataFrame(global_scores)
            all_res.append(pd.DataFrame({'func_ind': func_ind, 'n': n, 'B': B, 'feature': feature_names,
                                        'scores_mean': global_scores.mean().values, 
                                        'scores_std': (global_scores.std() / np.sqrt(rep)).values}))

    all_res = pd.concat(all_res)
    return all_res



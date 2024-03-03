import time, datetime, yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns


def train_model(learning_data, feature_names, target_name, model_type, hyperparameters, return_test=False, seed=1):
    # Split data
    test = learning_data.sample(frac=0.3, random_state=seed)
    train = learning_data.drop(test.index, axis=0)

    model_map = {'xgb': xgb.XGBRegressor,
                 'rf': xgb.XGBRFRegressor}

    model = model_map[model_type](**hyperparameters)
    model.fit(train[feature_names], train[target_name])
    model_scores = {'baseline score': np.round(model.score(test[feature_names],
                                                           np.full(len(test[target_name]), np.mean(train[target_name]))), 3),
                    'train score': np.round(model.score(train[feature_names], train[target_name]), 3),
                    'test score': np.round(model.score(test[feature_names], test[target_name]), 3)}
    if return_test:
        return model, model_scores, test
    return model, model_scores


def calc_efficiency(uppers, lowers):
    n = np.size(uppers)
    Rn_alpha = 1 - (1 / (n * (n - 1))) * np.sum(uppers - lowers)
    return (1 - Rn_alpha)


def calc_true_rank(true_scores):
    p = np.size(true_scores)
    lowers = np.full(p, 1) # 1 for all
    uppers = np.full(p, p) # p for all
    for j in range(p):
        lowers[j] += np.sum(true_scores[j] > np.delete(true_scores, j))
        uppers[j] -= np.sum(true_scores[j] < np.delete(true_scores, j))
    return lowers, uppers


def calc_coverage(true_scores, uppers, lowers):
    true_rank_lowers, true_rank_uppers = calc_true_rank(true_scores)
    coverage = np.size(np.where((lowers <= true_rank_lowers) & (true_rank_uppers <= uppers))[0]) / np.size(true_scores)
    return coverage, 1 if coverage == 1 else 0


def get_ranks_measures(true_score, rank_CI):
    res = {}
    res['efficiency'] = calc_efficiency(rank_CI['U'], rank_CI['L'])
    coverage, simul_coverage = calc_coverage(true_score, rank_CI['U'], rank_CI['L'])
    res['coverage'] = coverage
    res['simultaneous_coverage'] = simul_coverage
    return res


def calc_mean_measures(all_measures):
    # ranks measures of one method
    if type(all_measures) == list:
        mean_measures = pd.DataFrame(all_measures).mean(axis=0)
        mean_measures['efficiency_std'] = np.std(pd.DataFrame(all_measures)['efficiency'], ddof=1)
        return mean_measures
    # ranks measures of multiple methods
    mean_measures = {}
    for key, value in all_measures.items():
        mean_measures[key] = pd.DataFrame(value).mean(axis=0)
        mean_measures[key]['efficiency_std'] = np.std(pd.DataFrame(value)['efficiency'], ddof=1)
    return pd.DataFrame(mean_measures).T


def calc_time(start_time):
    seconds = time.time() - start_time
    return(str(datetime.timedelta(seconds=seconds)))


def L_U_to_CI(ranking):
    return ranking.apply(lambda x: list(x), axis=1)


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file: {e}")
            return None


def plot_corr(corr):
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(240, 10, n=20)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
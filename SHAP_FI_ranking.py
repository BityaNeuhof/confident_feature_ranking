import time, os, argparse
import numpy as np
from utils.general import calc_time, load_yaml_file
from experiments.simulated_data.low_variance_FI.SHAP.feature_importance_correctness_simulation import run_ranking_fi_simulation 


if __name__ == "__main__":
    np.random.seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    params = load_yaml_file(args.config_file)

    start_time = time.time()

    print('start ranking feature importance SHAP simulation at', time.strftime('%D %T', time.localtime(start_time)))
    res_imp_simulation = run_ranking_fi_simulation(params)

    out_directory = 'results/low_variance_FI/SHAP/'
    os.makedirs(out_directory, exist_ok=True)

    res_imp_simulation.to_csv(out_directory + 'feature_importance_ranks_measures_SHAP.csv', index=False)
    print('end ranking feature importance SHAP simulation. duration:', calc_time(start_time))
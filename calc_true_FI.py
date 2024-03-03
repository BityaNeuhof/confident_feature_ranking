import time, os, argparse
import numpy as np
from utils.general import calc_time, load_yaml_file
from experiments.simulated_data.low_variance_FI.SHAP.true_importance_all_models import true_importance_all_models



if __name__ == "__main__":
    np.random.seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    params = load_yaml_file(args.config_file)
    start_time = time.time()
    print('start calculate true feature importance at', time.strftime('%D %T', time.localtime(start_time)))
    res_true_imp = true_importance_all_models(params)

    out_directory = 'results/low_variance_FI/SHAP/'
    os.makedirs(out_directory, exist_ok=True)

    np.save(out_directory + 'true_feature_importance_all_models.npy', res_true_imp)
    print('end calculate true feature importance. duration:', calc_time(start_time))




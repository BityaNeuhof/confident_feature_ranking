import time, os, argparse
import numpy as np
from utils.general import calc_time, load_yaml_file
from experiments.simulated_data.low_variance_FI.SHAP.train_all_models import train_all_models


if __name__ == "__main__":
    np.random.seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    train_params = load_yaml_file(args.config_file)
    start_time = time.time()
    print('start train all models at', time.strftime('%D %T', time.localtime(start_time)))
    models_scores = train_all_models(train_params)

    out_directory = 'results/low_variance_FI/SHAP/'
    os.makedirs(out_directory, exist_ok=True)

    models_scores.to_csv(out_directory + 'models_scores.csv', index=False)
    print('end train all models. duration:', calc_time(start_time))

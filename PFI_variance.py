import time, os, argparse
import numpy as np
import pandas as pd
from utils.general import calc_time, load_yaml_file
from experiments.simulated_data.low_variance_FI.PFI.feature_importance_variance_simulation import run_pfi_base_to_global_variance_simulation


if __name__ == "__main__":
    np.random.seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    params = load_yaml_file(args.config_file)

    start_time = time.time()
    print('start PFI base-to-global variance simulation at', time.strftime('%D %T', time.localtime(start_time)))
    all_res = run_pfi_base_to_global_variance_simulation(**params)

    out_directory = 'results/low_variance_FI/PFI/'
    os.makedirs(out_directory, exist_ok=True)

    all_res.to_csv(out_directory + 'base_to_global_std.csv', index=False)
    print('end PFI base-to-global variance simulation. duration:', calc_time(start_time))
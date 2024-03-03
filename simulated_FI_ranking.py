import time, os, argparse
import numpy as np
from experiments.simulated_data.simulated_FI.simulated_FI_correctness_simulation import run_ranking_comparison_simulation
from utils.general import calc_time, load_yaml_file

if __name__ == "__main__":
    np.random.seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    params = load_yaml_file(args.config_file)

    out_suffix = params['corr_type'][0] + '_' + str(params['p'][0]) + '_' + time.strftime("%Y%m%d-%H%M%S")

    start_time = time.time()
    print('start ranking comparison correctness simulation at', time.strftime('%D %T', time.localtime(start_time)))
    res = run_ranking_comparison_simulation(params)

    out_directory = 'results/simulated_FI/'
    os.makedirs(out_directory, exist_ok=True)

    res.to_csv(out_directory + 'simulated_importance_ranks_measures_' +  out_suffix + '.csv', index=False)
    print('end ranking comparison correctness simulation. duration:', calc_time(start_time))
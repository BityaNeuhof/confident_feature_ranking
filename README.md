# Confident Feature Ranking

This is the code for our ranking method, experiments, and visualizations, as presented in our paper: Confident Feature Ranking.



### Packages reuirements:

The detailed list of packages (including versions) is available at the [requirements](requirements.txt) file.
We used a Python venv to run all the experiments.

-------------------------



### Run experiments - simulated data:

#### Ranking method comparison:

Without correlations:

```python simulated_FI_ranking.py ./jobs/simulated_FI_ranking_no_corr.yaml```

With correlations:

```python simulated_FI_ranking.py ./jobs/simulated_FI_ranking_corr.yaml```


-----------------

#### SHAP ranking measures:

Train models:

```python train_models.py ./jobs/train_models.yaml```

Calculate the true global FI values:

```python calc_true_FI.py ./jobs/calc_true_FI.yaml```

Compute ranking measures for SHAP:

```python SHAP_FI_ranking.py ./jobs/SHAP_FI_ranking.yaml```


-----------------

#### PFI variance analysis:

```python PFI_variance.py ./jobs/PFI_variance.yaml```


-----------------

#### Example of non-normal base FI values:

[Non-normal example notebook](./experiments/simulated_data/low_variance_FI/SHAP/shap_non_normal.ipynb)


-----------------

### Plot results - simulated data:

[Plot notebook](plot_simulations.ipynb)


-----------------


### Run experiments - real data:

* [Bike notebook](./experiments/real_data/bike_simulation.ipynb)
* [COMPAS notebook](./experiments/real_data/compas_simulation.ipynb)
* [Nomao notebook (High-dimensional example)](./experiments/real_data/nomao_simulation.ipynb)



-----------------



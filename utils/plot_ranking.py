import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.general import L_U_to_CI
from scipy.stats import rankdata


def plot_ranks(ranks, globals, feature_names, delta=0.2, ax=None, plot_globals=True, colors_map={}):
    p = len(feature_names)
    ranks_plot = pd.DataFrame()
    ranks_plot['feature'] = feature_names
    ranks_plot['globals_order'] = (rankdata(globals)).astype(int)
    ranks_plot['CI'] = L_U_to_CI(ranks)

    if not colors_map:
        colors = sns.color_palette(n_colors=p)
        colors_map = {feature: colors[i] for i, feature in enumerate(feature_names)}
    
    ranks_plot = ranks_plot.sort_values(by='globals_order', ignore_index=True)    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    for i, row in ranks_plot.iterrows():
        ax.hlines(y=row['globals_order'], xmin=row['CI'][0] - delta, xmax=row['CI'][1] + delta, 
                  linewidths=8, colors=colors_map[row['feature']])
        ax.errorbar(row['CI'][0] - delta, row['globals_order'], yerr=0.3, ecolor='k', elinewidth=2)
        ax.errorbar(row['CI'][1] + delta, row['globals_order'], yerr=0.3, ecolor='k', elinewidth=2)
        ax.vlines(row['globals_order'], 0, 1, transform=ax.get_xaxis_transform(), ls='--', color='k', alpha=0.2)

        if plot_globals:
            ax.plot(row['globals_order'], row['globals_order'], '^', markersize=8, c='k')
    
    ax.set_xticks(range(1, p + 1))
    ax.set_yticks(range(1, p + 1))
    ax.set_xticklabels(range(1, p + 1))
    ax.set_yticklabels(ranks_plot['feature'])
    ax.set_xlabel('Rank')
    ax.set_ylabel('Feature')
    ax.set_ylim(0)
    return fig
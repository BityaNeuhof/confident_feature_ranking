import numpy as np
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, ListVector
numpy2ri.activate()
r_icranks = importr('ICRanks')


def calc_icranks(means, stds, means_ord, method='Tukey', alpha=0.05):
    r_res = np.array(r_icranks.ic_ranks(means[means_ord], stds[means_ord], Method=method, alpha=alpha,
                                        control=ListVector({'trace':False, 'SwapPerm':False})))
    lowers = r_res[0,:][means_ord.argsort()]
    uppers = r_res[1,:][means_ord.argsort()]
    return pd.DataFrame([lowers, uppers], dtype=int).T.rename(columns={0: 'L', 1:'U'})
# ----------------------------------------------------------------------
# 1. Code for test Nie's algorithm of solving trace ratio
#     "Trace Ratio Criterion for Feature Selection 2008"
# 2. Naive approach for the trace ratio criterion
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sys

def nie_alg(dig_B, dig_E, m):
    d = dig_B.shape[0]

    tmp_p = np.zeros([d])
    p = np.copy(tmp_p) # initial the subset
    p[range(m)] = 1
    
    lam_old = -1  # lambda is greater than 0, so this is enough
    lam_ = np.dot(p,dig_B)/np.dot(p,dig_E)
    while 1:
        score_2 = dig_B - lam_*dig_E
        order = np.argsort(score_2)[::-1]
        p = np.copy(tmp_p)
        p[order[range(m)]] = 1
        lamb_old = lam_
        lam_ = np.dot(p,dig_B)/np.dot(p,dig_E)
        if abs(lam_ - lamb_old) < 0.001:
            break
    return np.array(np.where(p==1)).tolist()[0], lam_


def naive_trace_ratio(dig_B, dig_E, m):
    # set nan to -infinity
    # B_E_ratio[np.isnan(B_E_ratio)] = -np.inf
    order = np.argsort(dig_B/dig_E)[::-1]

    # the ratio selected by the naive method
    selected_features = order[0:m]
    ratio = sum(dig_B[selected_features])/sum(dig_E[selected_features])
    return selected_features, ratio


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: trace_raio_Nie.py m")
        print("  m  The number of features you want to selected")
        sys.exit(-1)
    m = int(sys.argv[1])


    # --------------------------------------------------
    # Main: 
    # --------------------------------------------------
    # Load the dataset
    _data = np.load('orl_data.npz')
    dig_B = _data['Bii']
    dig_E = _data['Eii']
    # m = 10
    print nie_alg(dig_B, dig_E, m)        


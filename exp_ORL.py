# ----------------------------------------------------------------------
# Experiment for ORL dataset
# ----------------------------------------------------------------------
# This code will include these parts: 
# [DONE] 2. Use 1NN to give the test accuracy, run 20 times
#             a. naive approach
#             b. subset approach
#             c. MIO approach
# ----------------------------------------------------------------------

import numpy as np
from random import shuffle
import heapq
import cplex
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# import functions
from data_orl_create import data_orl_random_split, data_orl_add_constraints
from function_benchmark import nie_alg, naive_trace_ratio
from function_mip import setupproblem_basic, eliminate_features



def fit_1NN(train_x, train_y, test_x, test_y, features):
    tmp_train_x = train_x[features,:].T
    tmp_test_x = test_x[features, :].T
    neigh = KNeighborsClassifier(n_neighbors=1)
    predict_y = neigh.fit(tmp_train_x, train_y) 
    predict_y = np.array(predict_y)
    test_y = np.array(test_y)
    error_num =  np.where(test_y - neigh.predict(tmp_test_x)!=0)[0].shape[0]
    return error_num


# These are the control parameters
# ==============================
loop_num = 20
threshold = 0.98
m_list = xrange(10,110,10)
# m_list = [10,20,30]
# ==============================

# things to record
# ==============================
run_time = []
error_naive = []
error_nie = []
error_mio = []
# ==============================

d = 4096
var_p = ['p'+str(i) for i in xrange(1,d+1)]
var_l = ['l'+str(i) for i in xrange(1,d+1)]
var_lambda = ['lambda']

for loop_i in range(loop_num):
    print 'loop ' + str(loop_i) + '...'
    mio_monot = 10000 # the bound of monotonic property, set to largest each time
    train_x, train_y, test_x, test_y, dig_B, dig_E = data_orl_random_split()
    cons_var, cons_coef, cons_rhs, ignore_list =  data_orl_add_constraints(threshold)
    error_m_naive = []
    error_m_nie = []
    error_m_mio = []
    for m in m_list:
        # naive approach
        features_naive, naive_ratio = naive_trace_ratio(dig_B, dig_E, m)
        error_m_naive.append(fit_1NN(train_x, train_y, test_x, test_y, features_naive))

        # nie approach
        features_nie, nie_ratio = nie_alg(dig_B,dig_E, m)
        error_m_nie.append(fit_1NN(train_x, train_y, test_x, test_y, features_nie))

        # the mio approach
        U_m = min([max(dig_B/dig_E), nie_ratio, sum(heapq.nlargest(m, dig_B))/sum(heapq.nsmallest(m, dig_E)), mio_monot])
        L_m = max([min(dig_B/dig_E), sum(heapq.nsmallest(m, dig_B))/sum(heapq.nlargest(m, dig_E))])
        c = cplex.Cplex()
        setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m, cons_var, cons_coef, cons_rhs)
        print 'eliminate '+ str(eliminate_features(c, d, dig_B, dig_E, U_m, ignore_list)) + '  variables'
        time = c.get_time()
        c.solve()
        run_time.append(c.get_time() - time)
        sol = c.solution
        mio_monot = sol.get_objective_value()

        features_mio = np.where(np.array(sol.get_values(var_p))>=0.5)[0]
        error_m_mio.append(fit_1NN(train_x, train_y, test_x, test_y, features_mio))
    error_naive.append(error_m_naive)
    error_nie.append(error_m_nie)
    error_mio.append(error_m_mio)
    
print np.mean(1 - np.matrix(error_naive)/float(len(test_y)),axis=0)
print np.std(1 - np.matrix(error_naive)/float(len(test_y)),axis=0)

print np.mean(1 - np.matrix(error_nie)/float(len(test_y)),axis=0)
print np.std(1 - np.matrix(error_nie)/float(len(test_y)),axis=0)

print np.mean(1 - np.matrix(error_mio)/float(len(test_y)),axis=0)
print np.std(1 - np.matrix(error_mio)/float(len(test_y)),axis=0)

np.savez('./data/orl_face/exp_orl_error', error_naive = error_naive, error_nie = error_nie, error_mio = error_mio, run_time = run_time)

# --------------------------------------------------
# plot of the selected features with the last selection


# --------------------------------------------------
# for a specific process, show how the elimination process works, how many new constraints and features are involved




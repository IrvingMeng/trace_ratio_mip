# ----------------------------------------------------------------------
# Experiment 1
# ----------------------------------------------------------------------
# This code will include these parts: 
# [DONE] 1. function for pre-process the data and give a set of b, e
#             a. Here we duplicate each features to show redundancy
# [TODO] 2. Use 1NN to give the test accuracy, run 20 times
#             a. naive approach
#             b. subset approach
#             c. MIO approach
# [TODO] 3. report the running time for the MIO approach
# ----------------------------------------------------------------------

import numpy as np
from random import shuffle
import heapq
import cplex
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier

# import functions
from function_benchmark import nie_alg
from function_benchmark import naive_trace_ratio
from function_mip import setupproblem_UCI



def generate_one_sample(data, targets):
    """
    Code for generate randomly split the data to training/testing
    --------------------------------------------------
    Input: @ data, targets
    Outputs: $ train_x, train_y, test_x, test_y
    """
    unique_y = np.unique(targets)
    K = len(unique_y)
    d, n = data.shape
    choose_list = []
    for k in range(K):
        choose_list.append(np.where(targets == k))
    # split the train/test data
    n_k = []
    mu_k = []
    Sigma_k = []
    for k in range(K):    
        tmp_list = choose_list[k][0]
        shuffle(tmp_list)
        tmp_n = len(tmp_list)
        tmp_train_x = data[:,tmp_list[0:int(tmp_n/2)]]
        tmp_train_y = targets[tmp_list[0:int(tmp_n/2)]]
        tmp_test_x = data[:,tmp_list[int(tmp_n/2)+1:-1]]
        tmp_test_y = targets[tmp_list[int(tmp_n/2)+1:-1]]
  
        n_k.append(tmp_train_x.shape[1])
        mu_k.append(np.mean(tmp_train_x,axis=1))
        Sigma_k.append(np.cov(tmp_train_x))
    
        if k == 0:
            train_x = tmp_train_x
            train_y = tmp_train_y
            test_x = tmp_test_x
            test_y = tmp_test_y
        else:
            train_x = np.concatenate([train_x, tmp_train_x],axis=1)
            train_y = np.concatenate([train_y, tmp_train_y])
            test_x = np.concatenate([test_x, tmp_test_x],axis=1)
            test_y = np.concatenate([test_y, tmp_test_y])

    # calculate the two matrix
    mu = np.zeros([d,1])
    for k in range(K):
        mu = mu+n_k[k]*mu_k[k] 
        mu = mu/sum(n_k)


    B = np.zeros([d,d])
    E = np.zeros([d,d])
    for k in range(K):
        E = E + n_k[k]*Sigma_k[k]
        B = B + n_k[k]*np.outer(mu_k[k]-mu,mu_k[k]-mu)

    dig_E = []
    dig_B = []
    for i in range(d):
        dig_B.append(B[i,i])
        dig_E.append(E[i,i])
    dig_B = np.array(dig_B)
    dig_E = np.array(dig_E)            
    return dig_B, dig_E, train_x, list(train_y), test_x, list(test_y), 

def fit_1NN(train_x, train_y, test_x, test_y, features):
    tmp_train_x = train_x[features,:].T
    tmp_test_x = test_x[features, :].T
    neigh = KNeighborsClassifier(n_neighbors=1)
    predict_y = neigh.fit(tmp_train_x, train_y) 
    predict_y = np.array(predict_y)
    test_y = np.array(test_y)
    error_num =  np.where(test_y - neigh.predict(tmp_test_x)!=0)[0].shape[0]
    return error_num




def data_ionosphere():
    """
    1. load the ionosphere data, acutual size (33, 351) (row 2 is deleted)
    2. Each feature is duplicated once
    """
    data =  np.genfromtxt('./data/ionosphere/ionosphere.data', delimiter=',', dtype=None)
    Data_X = []
    Data_Y = []
    for i in range(len(data)):
        Data_X.append(list(data[i])[0:-1])
        Data_Y.append(data[i][-1])
    Data_X = np.matrix(Data_X).T
    Data_X = np.delete(Data_X, (1), axis=0)  # delete the row 2 for it's all zero
    Data_X = np.vstack((Data_X, Data_X, Data_X)) # duplicate once
    Data_Y = [y=='g' for y in Data_Y]
    Data_Y = np.array(map(int, Data_Y))
    cons_var = []
    cons_coef = []
    cons_rhs = []
    for i in range(33):
        cons_var.append([i, 33+i, 66+i])
        cons_coef.append([1,1,1])
        cons_rhs.append([1])
    return Data_X, Data_Y, cons_var, cons_coef, cons_rhs



Data_X, Data_Y, cons_var, cons_coef, cons_rhs = data_ionosphere()

d, n = Data_X.shape
loop_num = 20
m = 8 # select feature number \approx d*0.25

error_list_naive = []
error_list_nie = []
error_list_mio = []
var_p = ['p'+str(i) for i in xrange(1,d+1)]
var_l = ['l'+str(i) for i in xrange(1,d+1)]
var_lambda = ['lambda']

for loop_i in range(loop_num):
    print 'loop ' + str(loop_i) + '...'
    error_one_loop_naive = [] # this store the accuracy for each splits
    error_one_loop_nie = [] # this store the accuracy for each splits
    dig_B, dig_E, train_x, train_y, test_x, test_y =  generate_one_sample(Data_X, Data_Y)
    
    # naive approach
    features_naive, naive_ratio = naive_trace_ratio(dig_B, dig_E, m)
    error_list_naive.append(fit_1NN(train_x, train_y, test_x, test_y, features_naive))

    # nie approach
    features_nie, nie_ratio = nie_alg(dig_B,dig_E, m)
    error_list_nie.append(fit_1NN(train_x, train_y, test_x, test_y, features_nie))
    # the mio approach
    M = min([max(dig_B/dig_E), nie_ratio, sum(heapq.nlargest(m, dig_B))/sum(heapq.nsmallest(m, dig_E))])
    c = cplex.Cplex()
    setupproblem_UCI(c, m, d, dig_B, dig_E, M, cons_var, cons_coef, cons_rhs)
    c.solve()
    sol = c.solution
    features_mio = np.where(np.array(sol.get_values(var_p))>=0.5)[0]

    error_list_mio.append(fit_1NN(train_x, train_y, test_x, test_y, features_mio))

    
print np.mean(1 - np.array(error_list_naive)/float(len(test_y)))
print np.std(1 - np.array(error_list_naive)/float(len(test_y)))

print np.mean(1 - np.array(error_list_nie)/float(len(test_y)))
print np.std(1 - np.array(error_list_nie)/float(len(test_y)))

print np.mean(1 - np.array(error_list_mio)/float(len(test_y)))
print np.std(1 - np.array(error_list_mio)/float(len(test_y)))



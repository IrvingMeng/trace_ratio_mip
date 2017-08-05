# ----------------------------------------------------------------------
# Experiment for ORL dataset
# ----------------------------------------------------------------------
# This code will include these parts: 
# A. [DONE] Compare the select  features when m=100, shown in a picture
# B. [DONE] Report how many features and constraints are added
# C. [DONE] With the increasing of m, report
#          [] 1. the running time v.s. without using the monotonic property
#          [] 2. Running time v.s. without using our framework at all
#          [] 3. number of eliminated features
# D. [TODO] Compare the running time of whether use the elimination process
#          [] With respect of m
# ----------------------------------------------------------------------
import numpy as np
import heapq
import cplex
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.datasets import fetch_olivetti_faces


# import functions
from data_orl_create import data_orl_add_constraints
from function_benchmark import nie_alg, naive_trace_ratio
from function_mip import setupproblem_basic, eliminate_features

d = 4096
var_p = ['p'+str(i) for i in xrange(1,d+1)]
var_l = ['l'+str(i) for i in xrange(1,d+1)]
var_lambda = ['lambda']

_data = np.load('./data/orl_face/orl_data.npz')
dig_B = _data['dig_B']
dig_E = _data['dig_E']
cons_var, cons_coef, cons_rhs, ignore_list =  data_orl_add_constraints(0.98)




# ------------------------------
# A
# this part produce a image to show the selected features
# can consider change threhsold
m = 200
features_naive, naive_ratio = naive_trace_ratio(dig_B, dig_E, m)
features_nie, nie_ratio = nie_alg(dig_B,dig_E, m)
mio_monot = 3 # the bound of monotonic property, set to largest each time    
# the mio approach
U_m = min([max(dig_B/dig_E), nie_ratio, sum(heapq.nlargest(m, dig_B))/sum(heapq.nsmallest(m, dig_E)), mio_monot])
L_m = max([min(dig_B/dig_E), sum(heapq.nsmallest(m, dig_B))/sum(heapq.nlargest(m, dig_E))])
c = cplex.Cplex()
setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m, cons_var, cons_coef, cons_rhs)
print 'eliminate '+ str(eliminate_features(c, d, dig_B, dig_E, U_m, ignore_list)) + '  variables'
c.solve()
sol = c.solution
features_mio = np.where(np.array(sol.get_values(var_p))>=0.5)[0]


# >>> features_mio
# array([   9,   11,   14,   18,   22,   26,   30,   35,   38,   49,   51,
#          53,   55,   56,   72,   74,   76,   80,   96,  137,  142,  146,
#         149,  153,  157,  163,  165,  167,  181,  183,  200,  203,  205,
#         207,  210,  213,  216,  219,  223,  225,  228,  230,  249,  265,
#         266,  268,  272,  277,  280,  282,  284,  286,  288,  290,  292,
#         344,  346,  348,  350,  352,  354,  356,  412,  414,  416,  418,
#         419,  481,  522,  523,  524,  544,  545,  546,  585,  586,  587,
#         588,  589,  844,  845,  846,  848,  849,  907,  908,  909, 1293,
#        1612, 1614, 1674, 1675, 1677, 1740, 2060, 2187, 2250, 2377, 2440,
#        2505])
# >>> features_naive
# array([148, 150, 149, 151,  19,  83,  84,  85, 147,  87,  86,  20,  18,
#         88, 152,  23,  82, 215, 153, 217, 214,  21,  24, 216, 146,  89,
#         25,  22, 145,  26,  17, 218,  81, 154, 281, 213,  27,  90,  80,
#         91,  92, 144,  16,  28,  14,  13,  15, 212,  79, 282, 143, 155,
#        142,  78, 219,  93,  77, 211,  12,  29, 157, 141, 156,  76, 280,
#        283,  30,  94,  11, 210, 208, 158, 220,  75,  31, 140,  96,  32,
#        209,  33, 221, 206, 222,  95, 139, 279, 207, 908,  34, 284,  97,
#        160, 346, 161, 159, 278, 286, 223, 225,  10])
# >>> features_nie
# [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 279, 280, 281, 282, 283, 284, 286, 287, 289, 346]


# --------------------------------------------------
# This part is used for B, C
cons_var, cons_coef, cons_rhs, ignore_list =  data_orl_add_constraints(0.98)
print len(cons_rhs) # of new constraints
print len(ignore_list) # of involved featues
m_list = range(10,110,10) + range(110,210,20) +  range(210,510,50)  +range(510, 1000, 100) + [1000]# 27 values to 1000


num_eli = []
run_time = []
mio_monot = 100
for m in m_list:
    # nie approach
    print "HERE"
    print m
    print 
    features_nie, nie_ratio = nie_alg(dig_B,dig_E, m)
    # the mio approach
    U_m = min([max(dig_B/dig_E), nie_ratio, sum(heapq.nlargest(m, dig_B))/sum(heapq.nsmallest(m, dig_E)), mio_monot])
    L_m = max([min(dig_B/dig_E), sum(heapq.nsmallest(m, dig_B))/sum(heapq.nlargest(m, dig_E))])
    c = cplex.Cplex()
    setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m, cons_var, cons_coef, cons_rhs)
    num_eli.append(eliminate_features(c, d, dig_B, dig_E, U_m, ignore_list))
    time = c.get_time()
    c.solve()
    run_time.append(c.get_time() - time)
    sol = c.solution
    mio_monot = sol.get_objective_value()



# without the monotonic property
m = 1000
features_nie, nie_ratio = nie_alg(dig_B,dig_E, m)
# the mio approach
U_m = min([max(dig_B/dig_E), nie_ratio, sum(heapq.nlargest(m, dig_B))/sum(heapq.nsmallest(m, dig_E))])
L_m = max([min(dig_B/dig_E), sum(heapq.nsmallest(m, dig_B))/sum(heapq.nlargest(m, dig_E))])
c = cplex.Cplex()
time = c.get_time()
setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m, cons_var, cons_coef, cons_rhs)
num_eli.append(eliminate_features(c, d, dig_B, dig_E, U_m, ignore_list))
c.solve()
run_time.append(c.get_time() - time)


# doesn't use the elimination
c = cplex.Cplex()
time = c.get_time()
setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m, cons_var, cons_coef, cons_rhs)
c.solve()
run_time.append(c.get_time() - time)

np.savez('./data/orl_face/orl_data_mon_time', m_list = m_list, num_eli = num_eli, run_time = run_time)

# the saved data 1:27 of num_eli, run_time correspond to m_list, 28 of num_eli, run_time uses no monotonic property, 29 of run_time doesn't eliminate the variables

# ----------
# plot the results
import seaborn as sns    
plt.rc('text', usetex=True)
plt.plot(m_list, num_eli[0:-1],'bo-')
plt.xlabel(r'\bf Target number of features $\mathbf{m}$',fontsize = 15)
plt.ylabel(r'\bf Number of eliminated variables',fontsize = 15)
plt.show()

plt.plot(m_list, run_time[0:-2])
plt.xlabel(r'\bf Target number of features $\mathbf{m}$',fontsize = 15)
plt.ylabel(r'\bf Running time (s)',fontsize = 15)
plt.show()


# ------------------------------
# Part D
print "Let's go!'"
m_list = range(400,3050,50)

num_D_eli = []  # D here represents part D
time_D_elimination = []
time_D_no_elimination = []
mio_monot = 3.4 # m = 200
for m in m_list:
    features_nie, nie_ratio = nie_alg(dig_B,dig_E, m)
    U_m = min([max(dig_B/dig_E), nie_ratio, sum(heapq.nlargest(m, dig_B))/sum(heapq.nsmallest(m, dig_E)), mio_monot])
    L_m = max([min(dig_B/dig_E), sum(heapq.nsmallest(m, dig_B))/sum(heapq.nlargest(m, dig_E))])


    c = cplex.Cplex()
    c.set_results_stream(None) # stop printing 
    time = c.get_time()
    setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m, cons_var, cons_coef, cons_rhs)
    num_D_eli.append(eliminate_features(c, d, dig_B, dig_E, U_m, ignore_list))
    c.solve()
    time_D_elimination.append(c.get_time() - time)

    sol = c.solution
    mio_monot = sol.get_objective_value()
    
    print "Eliminate  " + str(num_D_eli[-1]) + "  out of " + str(m) + " features"
    print "         running time: " + str(time_D_elimination[-1])
    
    c = cplex.Cplex()
    c.set_results_stream(None) # stop printing 
    time = c.get_time()
    setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m, cons_var, cons_coef, cons_rhs)
    c.solve()
    time_D_no_elimination.append(c.get_time() - time)

    print "         running time(no elimination): " + str(time_D_no_elimination[-1])



# print m_list
# print num_D_eli
# print time_D_elimination
# print time_D_no_elimination
# ----------
# plot the results
import seaborn as sns    
plt.rc('text', usetex=True)
plt.plot(m_list,time_D_no_elimination, 'r-o',linewidth = 3)
plt.plot(m_list,time_D_elimination,'b-*',linewidth = 3)
plt.xlabel(r'\bf number of features',fontsize = 15)
plt.ylabel(r'\bf Running time',fontsize = 15)
labels = (r'\bf without elinimation', r'\bf with elimination')
legend = plt.legend(labels, loc=0,fontsize = 12)
plt.xlim([400, 3000])
plt.show()


plt.plot(m_list, num_D_eli,'bo-')
plt.xlabel(r'\bf Target number of features $\mathbf{m}$',fontsize = 15)
plt.ylabel(r'\bf Number of eliminated variables',fontsize = 15)
plt.xlim([400, 3000])
plt.show()

np.savez('./data/orl_face/orl_no_elimination_or_not', m_list = m_list, num_D_eli = num_D_eli, time_D_elimination = time_D_elimination, time_D_no_elimination = time_D_no_elimination)
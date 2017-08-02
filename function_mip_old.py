# ----------------------------------------------------------------------
# Code for test the trace ratio problem
# ----------------------------------------------------------------------
# This code will include these parts: 
#        1. naive trace ratio(naive_trace_ratio)
#        2. set up the problem and solve it with mip, return the selected 
#           variables and running time(mip_trace_ratio)
#        3. with new constraints added, set up the problem and solve it
# ----------------------------------------------------------------------
import numpy as np
from random import shuffle
import cplex
from matplotlib import pyplot as plt
import seaborn as sns


def setupproblem(c, m, d, dig_B, dig_E, M_i,  cons_var=[], cons_coef = [], cons_rhs = []):
    # define the variable and bound
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    var_l = ['l'+str(i) for i in xrange(1,d+1)]
    var_lambda = ['lambda']
    c.variables.add(names=var_lambda, types="C", obj = [1])
    c.variables.add(names=var_l, types=["C"]*d)
    c.variables.add(names=var_p, types=["B"]*d)
    
    # define the objective function
    c.objective.set_name("trace_ratio")
    c.objective.set_sense(c.objective.sense.maximize)


    U_bound = M_i  # eliminate the variables
    for i in range(d):
        if dig_B[i]/dig_E[i]>=U_bound:
            c.linear_constraints.add(lin_expr=[cplex.SparsePair([var_p[i]], [1])],
                                     senses=["E"],
                                     rhs=[1])    

    # add the constraints
    # sum p_i = m
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(var_p, [1]*d)],
                             senses=["E"],
                             rhs=[m])    

    # c.linear_constraints.add(lin_expr=[cplex.SparsePair(var_lambda, [1])],
    #                          senses=["G"],
    #                          rhs=[naive_trace_ratio(dig_E, dig_B, m)[1]])

    c.linear_constraints.add([cplex.SparsePair(var_l+var_lambda, [1]*d+[-m])],
                             senses=["E"],
                             rhs=[0])    # sum l_i = m lambda
    

    # sum p_i dig_B - sum l_i dig_E \leq 0
    thevars = []
    thecoefs = []
    for i in range(d):
        thevars.append(var_p[i])
        thecoefs.append(dig_B[i])
        thevars.append(var_l[i])
        thecoefs.append(-dig_E[i])
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                             senses=["E"],
                             rhs=[0])

    for i in range(d):
        # c.linear_constraints.add(lin_expr=
        #                          [cplex.SparsePair(var_lambda+ [var_l[i], var_p[i]], [-1, 1,M_i])],
        #                          senses=["L"],
        #                          rhs=[M_i])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair(var_lambda + [var_l[i]], [-1, 1])],
                                 senses=["L"],
                                 rhs=[0])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair(var_lambda+[var_l[i], var_p[i]], [1,-1,M_i])],
                                 senses=["L"],
                                 rhs=[M_i])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair([var_l[i], var_p[i]], [1,-M_i])],
                                 senses=["L"],
                                 rhs=[0])
        # c.linear_constraints.add(lin_expr=
        #                          [cplex.SparsePair([var_l[i], var_p[i]], [1,M_i])],
        #                          senses=["G"],
        #                          rhs=[0])


    for i in range(len(cons_var)):
        c.linear_constraints.add(lin_expr=
                                [cplex.SparsePair([var_p[k] for k in cons_var[i]], cons_coef[i])],
                                 senses=["L"],
                                 rhs=cons_rhs[i])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair([var_l[k] for k in cons_var[i]]+var_lambda, cons_coef[i]+[-1])],
                                 senses=["L"],
                                 rhs=[0])
        

def setupproblem_UCI(c, m, d, dig_B, dig_E, M_i,  cons_var=[], cons_coef = [], cons_rhs = []):
    # define the variable and bound
    # this is the process without elimating the varaibles
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    var_l = ['l'+str(i) for i in xrange(1,d+1)]
    var_lambda = ['lambda']
    c.variables.add(names=var_lambda, types="C", obj = [1])
    c.variables.add(names=var_l, types=["C"]*d)
    c.variables.add(names=var_p, types=["B"]*d)
    
    # define the objective function
    c.objective.set_name("trace_ratio")
    c.objective.set_sense(c.objective.sense.maximize)


    # U_bound = M_i  # eliminate the variables
    # for i in range(d):
    #     if dig_B[i]/dig_E[i]>=U_bound:
    #         c.linear_constraints.add(lin_expr=[cplex.SparsePair([var_p[i]], [1])],
    #                                  senses=["E"],
    #                                  rhs=[1])    

    # add the constraints
    # sum p_i = m
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(var_p, [1]*d)],
                             senses=["E"],
                             rhs=[m])    

    # c.linear_constraints.add(lin_expr=[cplex.SparsePair(var_lambda, [1])],
    #                          senses=["G"],
    #                          rhs=[naive_trace_ratio(dig_E, dig_B, m)[1]])

    c.linear_constraints.add([cplex.SparsePair(var_l+var_lambda, [1]*d+[-m])],
                             senses=["E"],
                             rhs=[0])    # sum l_i = m lambda
    

    # sum p_i dig_B - sum l_i dig_E \leq 0
    thevars = []
    thecoefs = []
    for i in range(d):
        thevars.append(var_p[i])
        thecoefs.append(dig_B[i])
        thevars.append(var_l[i])
        thecoefs.append(-dig_E[i])
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                             senses=["E"],
                             rhs=[0])

    for i in range(d):
        # c.linear_constraints.add(lin_expr=
        #                          [cplex.SparsePair(var_lambda+ [var_l[i], var_p[i]], [-1, 1,M_i])],
        #                          senses=["L"],
        #                          rhs=[M_i])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair(var_lambda + [var_l[i]], [-1, 1])],
                                 senses=["L"],
                                 rhs=[0])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair(var_lambda+[var_l[i], var_p[i]], [1,-1,M_i])],
                                 senses=["L"],
                                 rhs=[M_i])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair([var_l[i], var_p[i]], [1,-M_i])],
                                 senses=["L"],
                                 rhs=[0])
        # c.linear_constraints.add(lin_expr=
        #                          [cplex.SparsePair([var_l[i], var_p[i]], [1,M_i])],
        #                          senses=["G"],
        #                          rhs=[0])


    for i in range(len(cons_var)):
        c.linear_constraints.add(lin_expr=
                                [cplex.SparsePair([var_p[k] for k in cons_var[i]], cons_coef[i])],
                                 senses=["L"],
                                 rhs=cons_rhs[i])
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair([var_l[k] for k in cons_var[i]]+var_lambda, cons_coef[i]+[-1])],
                                 senses=["L"],
                                 rhs=[0])
        




def mip_trace_ratio(c, dig_E, dig_B, m, M_i):
    # -----
    # Build the model
    d = len(dig_E)
    setupproblem(c,m, d, dig_B, dig_E, M_i)
    c.solve()
    sol = c.solution
    print(sol.get_objective_value())
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    return np.where(np.array(sol.get_values(var_p))==1), sol.get_objective_value()





if __name__ == "__main__":

    # --------------------------------------------------
    # Part 1: Preprocess the data
    #      1. load the data
    #      2. build train and test 1:1 randomly
    #      3. Build the E and B matrix form the training data
    # --------------------------------------------------
    # Load the dataset
    _data = np.load('orl_data.npz')
    dig_B = _data['Bii']
    dig_E = _data['Eii']



    m = 10
    L_bound = naive_trace_ratio(dig_E, dig_B, m)[1]
    print L_bound
    # --------------------------------------------------
    # 3. Use cplex to solve the MIP (Problem) 
    # ----------define the variable and bound
    # ----------define the control_variable m, M_i
    # ----------add the constraints
    # --------------------------------------------------
    d = len(dig_E)
    var_lambda = ['lambda']
    var_l = ['l'+str(i) for i in xrange(1,d+1)]
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    m = 10
    c = cplex.Cplex()
    M_i = max(dig_B/dig_E)
    print mip_trace_ratio(c, dig_E, dig_B, m, M_i) 

    c = cplex.Cplex()
    print mip_trace_ratio_v2(c, dig_E, dig_B, m, M_i, L_bound) 



# _data = np.load('orl_data.npz')
# dig_B = _data['Bii']
# dig_E = _data['Eii']



# m = 10
# print naive_trace_ratio(dig_E, dig_B, m)

# d = len(dig_E)
# var_lambda = ['lambda']
# var_l = ['l'+str(i) for i in xrange(1,d+1)]
# var_p = ['p'+str(i) for i in xrange(1,d+1)]
# m = 10
# c = cplex.Cplex()
# M_i = max(dig_B/dig_E)
# print mip_trace_ratio(c, dig_E, dig_B, m, M_i) 

# c = cplex.Cplex()
# print mip_trace_ratio_v2(c, dig_E, dig_B, m, M_i, naive_trace_ratio(dig_E, dig_B, m-1)[1]) 
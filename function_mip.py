# ----------------------------------------------------------------------
# Code for test the trace ratio problem
# ----------------------------------------------------------------------
# This code includes: 
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





def setupproblem_basic(c, m, d, dig_B, dig_E, U_m, L_m = 0, cons_var=[], cons_coef = [], cons_rhs = []):
    """
    set up the basic mip model without eleminate the variables. 
    (Here maybe a class will be better. If needed, can switch to that later)
    --------------------------------------------------
    # this part should be defined in advance
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    var_l = ['l'+str(i) for i in xrange(1,d+1)]
    var_lambda = ['lambda']
    --------------------------------------------------
    Reture the cplex objective c
    """
    # add the variables
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    var_l = ['l'+str(i) for i in xrange(1,d+1)]
    var_lambda = ['lambda']
    c.variables.add(names=var_lambda, types="C", obj = [1])
    c.variables.add(names=var_l, types=["C"]*d)
    c.variables.add(names=var_p, types=["B"]*d)
    
    # define the objective function
    c.objective.set_name("trace_ratio")
    c.objective.set_sense(c.objective.sense.maximize)

    # add the constraints
    # ------------------------------
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
    # \sum p_i = m
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(var_p, [1]*d)],
                             senses=["E"],
                             rhs=[m])    
    # \sum l_i = m \lambda
    c.linear_constraints.add([cplex.SparsePair(var_l+var_lambda, [1]*d+[-m])],
                             senses=["E"],
                             rhs=[0])    
    for i in range(d):
        # l_i \leq \lambda 
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair(var_lambda + [var_l[i]], [-1, 1])],
                                 senses=["L"],
                                 rhs=[0])
        # l_i \leq U_m p_i
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair([var_l[i], var_p[i]], [1,-U_m])],
                                 senses=["L"],
                                 rhs=[0])
        # l_i \geq \lambda - U_m (1-p_i)
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair(var_lambda+[var_l[i], var_p[i]], [1,-1,U_m])],
                                 senses=["L"],
                                 rhs=[U_m])
    # \lambda \leq U_m
    c.linear_constraints.add([cplex.SparsePair(var_lambda, [1])],
                             senses=["L"],
                             rhs=[U_m])      
    # \lambda \geq L_m   
    c.linear_constraints.add([cplex.SparsePair(var_lambda, [1])],
                             senses=["G"],
                             rhs=[L_m])      
    # ------------------------------
        
    # additional constraints 
    for i in range(len(cons_var)):
        # constraints for p_i
        c.linear_constraints.add(lin_expr=
                                [cplex.SparsePair([var_p[k] for k in cons_var[i]], cons_coef[i])],
                                 senses=["L"],
                                 rhs=[cons_rhs[i]])
        # constraints for l_i
        c.linear_constraints.add(lin_expr=
                                 [cplex.SparsePair([var_l[k] for k in cons_var[i]]+var_lambda, cons_coef[i]+[-cons_rhs[i]])],
                                 senses=["L"],
                                 rhs=[0])
        

def eliminate_features(c, d, dig_B, dig_E, U_m, ignore_list):
    """
    U_m: the upper bound for elimination.
    ignore_list: the list that variables are included in other constraints
    """
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    var_l = ['l'+str(i) for i in xrange(1,d+1)]
    var_lambda = ['lambda']

    count = 0
    for i in range(d):
        if i in ignore_list:
            continue
        if dig_B[i]/dig_E[i]>=U_m:
            count = count + 1
            c.linear_constraints.add(lin_expr=[cplex.SparsePair([var_p[i]], [1])],
                                     senses=["E"],
                                     rhs=[1])            
            c.linear_constraints.add(lin_expr=[cplex.SparsePair([var_l[i]]+var_lambda, [1,-1])],
                                     senses=["E"],
                                     rhs=[0])     
    return count




def mip_trace_ratio(c, dig_E, dig_B, m, U_m):
    # -----
    # Build the model
    d = len(dig_E)
    setupproblem_basic(c,m, d, dig_B, dig_E, U_m)
    print eliminate_features_basic(c, d, dig_B, dig_E, U_m)
    c.solve()
    sol = c.solution
    print(sol.get_objective_value())
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    return np.where(np.array(sol.get_values(var_p))==1), sol.get_objective_value()





if __name__ == "__main__":
    # Load the dataset
    _data = np.load('./data/orl_face/orl_data.npz')
    dig_B = _data['dig_B']
    dig_E = _data['dig_E']
    d = len(dig_E)
    var_p = ['p'+str(i) for i in xrange(1,d+1)]
    var_l = ['l'+str(i) for i in xrange(1,d+1)]
    var_lambda = ['lambda']
    c = cplex.Cplex()
    U_m = max(dig_B/dig_E)
    m = 200
    # U_m = 5
    print mip_trace_ratio(c, dig_E, dig_B, m, U_m) 



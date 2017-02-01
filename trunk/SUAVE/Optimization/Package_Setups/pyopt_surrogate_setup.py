# surrogate_setup.py
#
# Created:  May 206, M. Vegh
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import pyOpt #use pyOpt to set up the problem
import numpy as np
from SUAVE.Core import Data
from SUAVE.Optimization import helper_functions as helper_functions

def pyopt_surrogate_setup(surrogate_function, inputs, constraints):
    #sets up a surrogate problem so it can be run by pyOpt
    

    #taken from initial optimization problem that you run
    ini              = inputs[:,1] # values
    bnd              = inputs[:,2] # Bounds
    scl              = inputs[:,3] # Scaling
    input_units      = inputs[:,-1] *1.0
    constraint_scale = constraints[:,3]
    constraint_units = constraints[:,-1]*1.0
    opt_problem      = pyOpt.Optimization('surrogate', surrogate_function)
    
    #constraints
    bnd_constraints    = helper_functions.scale_const_bnds(constraints)
    scaled_constraints = helper_functions.scale_const_values(constraints,bnd_constraints)
    constraints_out    = scaled_constraints*constraint_units
    scaled_inputs      = ini/scl
    x                  = scaled_inputs#*input_units
    
    print 'x_setup=', x
    #bound the input variables
    for j in range(len(inputs[:,1])):
        lbd = bnd[j][0]/(scl[j])#*input_units[j]
        ubd = bnd[j][1]/(scl[j])#*input_units[j]
        opt_problem.addVar('x%i' % j, 'c', lower = lbd, upper = ubd, value = x[j])
 
    #put in the constraints
    for j in range(len(constraints[:,0])):
        edge = constraints_out[j]
        if constraints[j][1]=='<':
            opt_problem.addCon('g%i' % j, type ='i', upper=edge)
        elif constraints[j][1]=='>':
            opt_problem.addCon('g%i' % j, lower=edge,upper=np.inf)
      
        elif constraints[j][1]=='>':
            opt_problem.addCon('g%i' % j, type='e', equal=edge)
      
    
    opt_problem.addObj('f')
    
    return opt_problem
 

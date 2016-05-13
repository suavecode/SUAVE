# vypy_surrogate_functions.py
#
# Created:  May 206, M. Vegh
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


from SUAVE.Core import Data
import helper_functions
from sklearn import svm

import pyOpt  
from read_optimization_outputs import read_optimization_outputs
from Optimization.Package_Setups.surrogate_setup import surrogate_problem

import numpy as np
import time


def build_svr_models(filename, base_inputs, constraint_inputs, kernel = 'rbf', C = 1E-4):
    bnd              = base_inputs[:,2] # Bounds
    scl              = base_inputs[:,3] # Scaling
    input_units      = base_inputs[:,-1] *1.0
    
    iterations, obj_values, inputs, constraints = read_optimization_outputs(filename, base_inputs, constraint_inputs)
    
    #now build surrogates based on these
    t1=time.time()

    # start a training data object
    clf             = svm.SVR(kernel=kernel, C=C)
    obj_surrogate   = clf.fit(inputs, obj_values) 
    constraints_surrogates = []
   
    #now do this for every constraint
    
    for j in range(len(constraints[0,:])):
        clf                  = svm.SVR(kernel=kernel, C=C)
        constraint_surrogate = clf.fit(inputs, constraints[:,j]) 
        constraints_surrogates.append(constraint_surrogate)
     
    t2=time.time()
    print 'time to set up = ', t2-t1
    surrogate_function    = surrogate_problem()
    surrogate_function.obj_surrogate          = obj_surrogate
    surrogate_function.constraints_surrogates = constraints_surrogates
    
    return obj_surrogate, constraints_surrogates, surrogate_function    
    
    



from SUAVE.Core import Data
import helper_functions
import pyKriging
import pyOpt  
from pyKriging.krige import kriging  
from read_optimization_outputs import read_optimization_outputs
from Package_Setups.surrogate_setup import surrogate_problem

import numpy as np
import time


def build_kriging_models(filename, base_inputs, constraint_inputs):

    iterations, obj_values, inputs, constraints = read_optimization_outputs(filename, base_inputs, constraint_inputs)

    print 'obj_values = ', obj_values
    print 'inputs=', inputs
    print 'constraints=', constraints
    
    #now build surrogates based on these
    t1=time.time()
  
    
    obj_surrogate = kriging(inputs, obj_values , name='simple')
    obj_surrogate.train()
    constraints_surrogates = []
   
    
    
    for j in range(len(constraints[0,:])):
        print 'j=', j
        constraint_surrogate = kriging(inputs, constraints[:,j] , name='simple')
        constraint_surrogate.train()
        constraints_surrogates.append(constraint_surrogate)
    t2=time.time()
    print 'time to set up = ', t2-t1
    surrogate_function    = surrogate_problem()
    surrogate_function.obj_surrogate          = obj_surrogate
    surrogate_function.constraints_surrogates = constraints_surrogates
    
    return obj_surrogate, constraints_surrogates, surrogate_function    
    
    



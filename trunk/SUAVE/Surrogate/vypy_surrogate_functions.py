# vypy_surrogate_functions.py
#
# Created:  May 2016, M. Vegh
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


from SUAVE.Core import Data
import VyPy
from VyPy.regression import gpr

from Surrogate_Problem import Surrogate_Problem

import numpy as np
import time


def build_gpr_models(obj_values, inputs, constraints, base_inputs):
    bnd              = base_inputs[:,2] # Bounds
    scl              = base_inputs[:,3] # Scaling
    input_units      = base_inputs[:,-1] *1.0
    
   
    #now build surrogates based on these
    t1=time.time()
    bounds = []
    for j in range(len(base_inputs[:,1])):
        lbd = bnd[j][0]*input_units[j]/(scl[j])
        ubd = bnd[j][1]*input_units[j]/(scl[j])
        bounds.append([lbd, ubd])
    bounds = np.array(bounds)

    # start a training data object
    Model                 = gpr.library.Gaussian(bounds, inputs, obj_values) #start training object
    obj_surrogate         = Data()
    obj_surrogate.predict = Model.predict_YI
   
   
    constraints_surrogates = []
   
    #now do this for every constraint
    
    for j in range(len(constraints[0,:])):
        Model                        = gpr.library.Gaussian(bounds, inputs, constraints[:,j])
        constraint_surrogate         = Data()
        constraint_surrogate.predict = Model.predict_YI
   
        constraints_surrogates.append(constraint_surrogate)
     
    t2=time.time()
    print 'time to set up = ', t2-t1
    surrogate_function    = Surrogate_Problem()
    surrogate_function.obj_surrogate          = obj_surrogate
    surrogate_function.constraints_surrogates = constraints_surrogates
    
    return obj_surrogate, constraints_surrogates, surrogate_function    
    
    



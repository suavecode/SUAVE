# svr_surrogate_functions.py
#
# Created:  May 206, M. Vegh
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


from SUAVE.Core import Data
from sklearn import svm
from Surrogate_Problem import Surrogate_Problem

import numpy as np
import time


def build_svr_models(obj_values, inputs, constraints, kernel = 'rbf', C = 1E5):
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
    surrogate_function                        = Surrogate_Problem()
    surrogate_function.obj_surrogate          = obj_surrogate
    surrogate_function.constraints_surrogates = constraints_surrogates
    
    return obj_surrogate, constraints_surrogates, surrogate_function    
    
    



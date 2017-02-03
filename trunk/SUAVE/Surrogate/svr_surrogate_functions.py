# svr_surrogate_functions.py
#
# Created:  May 2016, M. Vegh
# Modified:


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


from SUAVE.Core import Data
from sklearn import svm
from Surrogate_Problem import Surrogate_Problem

import numpy as np
import time


# ----------------------------------------------------------------------
#  svr_surrogate_functions
# ----------------------------------------------------------------------



def build_svr_models(obj_values, inputs, constraints, kernel = 'rbf', C = 1E5, epsilon =.01):
    #now build surrogates based on these
    t1=time.time()

    # start a training data object
    clf             = svm.SVR(kernel=kernel, C=C, epsilon = epsilon)
    obj_surrogate   = clf.fit(inputs, obj_values) 
    constraints_surrogates = []
   
    #now do this for every constraint
    
    for j in range(len(constraints[0,:])):
        clf                  = svm.SVR(kernel=kernel, C=C, epsilon = epsilon)
        constraint_surrogate = clf.fit(inputs, constraints[:,j]) 
        constraints_surrogates.append(constraint_surrogate)
     
    t2=time.time()
    print 'time to set up = ', t2-t1
    surrogate_function                        = Surrogate_Problem()
    surrogate_function.obj_surrogate          = obj_surrogate
    surrogate_function.constraints_surrogates = constraints_surrogates
    
    return obj_surrogate, constraints_surrogates, surrogate_function    
    
def check_svr_accuracy(x, data_inputs, data_outputs, imin = -1): #set up so you can frame as an optimization problem
    # x is inputs that you have option to optimize over
    #imin is index you want to leave out (default is last entry
    #use log base 10 inputs to find parameters
    Cval= 10**x[0]
    eps = 10**x[1]
    #prevents negative values

    y = []
    
    #omit one data point (by default the last one
    if imin == 0:
        data_inputs2 = data_inputs[imin+1:]
        data_outputs2 = data_outputs[imin+1:]
    else:
        data_inputs2 = np.vstack((data_inputs[:imin], data_inputs[imin+1:]))
        data_outputs2 = np.vstack((data_outputs[:imin], data_outputs[imin+1:]))
        
    for j in range (len(data_outputs[0,:])): #loop over data
        clf         = svm.SVR(C=Cval,  epsilon = eps)
        y_surrogate = clf.fit(data_inputs2, data_outputs2[:,j]) #leave out closest data point for surrogate fit
        y.append(y_surrogate.predict(data_inputs[-1,:])[0])
    y = np.array(y)
    y_real = data_outputs[imin,:]
    diff = (y_real-y)/y_real
    output= np.linalg.norm(diff)
    return output
        



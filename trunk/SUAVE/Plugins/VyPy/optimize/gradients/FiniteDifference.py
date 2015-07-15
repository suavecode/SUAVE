
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from VyPy.data import OrderedBunch as obunch
from VyPy.tools import atleast_2d_row, atleast_2d_col, array_type

from copy import deepcopy
import numpy as np
        
# ----------------------------------------------------------------------
#   Finite Difference Gradient
# ----------------------------------------------------------------------

class FiniteDifference(object):
    
    def __init__(self,function,step=1e-6):
        
        self.function = function
        self.step = step
        
        return
    
    def function(self,xs):
        
        nx = xs.shape[0]
        
        f = [0]*nx
        
        for i,x in xs:
            
            f[i] = self.function(x)
            
        return f
        
        
    def __call__(self,variables):
        
        step = self.step
        
        variables = deepcopy(variables)
        
        if not isinstance(variables,obunch):
            variables = obunch(variables)
            
        # arrayify variables
        values_init = variables.pack_array('vector')
        values_init = atleast_2d_row(values_init)
        nx = values_init.shape[1]
        
        # prepare step
        if not isinstance(step,(array_type,list,tuple)):
            step = [step] * nx
        step = atleast_2d_col(step)
        if not step.shape[1] == 1:
            step = step.T
        
        values_run = np.hstack([ values_init , 
                                 np.tile(values_init,[nx,1]) + np.diag(step) ])
            
        # run the function
        results = self.function(values_run)
        
        # pack results
        gradients_values = ( results[1:,:] - results[None,0,:] ) / step
        gradients_values = np.ravel(gradients_values)
        
        variables.unpack_array( values_init * 0.0 )
        
        gradients = deepcopy(results[0])
        for k in gradients.keys():
            gradients[k] = deepcopy(variables)
        
        gradients.unpack_array(gradients_values)
        
        return gradients
        
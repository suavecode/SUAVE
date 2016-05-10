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

def setup_surrogate_problem(surrogate_function, inputs, constraints):
    #inputs are inputs from a SUAVE Optimization case
    #constraints are constraints from a SUAVE Optimization

    #taken from initial optimization problem that you run
    names            = inputs[:,0] # Names
    ini              = inputs[:,1] # values
    bnd              = inputs[:,2] # Bounds
    scl              = inputs[:,3] # Scaling
    input_units      = inputs[:,-1] *1.0
    constraint_names = constraints[:,0]
    constraint_scale = constraints[:,3]
    constraint_units = constraints[:,-1]*1.0
    opt_problem      = pyOpt.Optimization('surrogate', surrogate_function)
    
    #constraints
    bnd_constraints    = helper_functions.scale_const_bnds(constraints)
    scaled_constraints = helper_functions.scale_const_values(constraints,bnd_constraints)
    constraints_out    = scaled_constraints*constraint_units

    
    scaled_inputs      = ini/scl
    
    x                  = scaled_inputs*input_units
    
    
    for j in range(len(inputs[:,1])):
        name = names[j]
        lbd = bnd[j][0]*input_units[j]/(scl[j])
        ubd = bnd[j][1]*input_units[j]/(scl[j])
        opt_problem.addVar(name, 'c', lower = lbd, upper = ubd, value = x[j])
    
    for j in range(len(constraints[:,0])):
        #only using inequality constraints, where we want everything >0
        name = constraint_names[j]
        edge = constraints_out[j]
        opt_problem.addCon(name, type ='i', lower=edge, upper=np.inf)
    
    opt_problem.addObj('f')
    
    return opt_problem, surrogate_function
 
class surrogate_problem(Data):
    def __defaults__(self):
        self.obj_surrogate = None
        self.constraints_surrogates = None
    
    def compute(self, x):
        f = self.obj_surrogate.predict(x)
        
        g = []
        for j in range(len(self.constraints_surrogates)):
            g.append(self.constraints_surrogates[j].predict(x))
        
        fail = 0
       
        return f, g, fail
        
    __call__ = compute
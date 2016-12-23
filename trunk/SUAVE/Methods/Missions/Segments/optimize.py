# optimize.py
# 
# Created:  Dec 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy.optimize as opt
import numpy as np

from SUAVE.Core.Arrays import array_type
from SUAVE.Core import Units
import copy

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

def converge_opt(segment,state):
    
    # pack up the array
    unknowns = state.unknowns.pack_array()
    
    # Have the optimizer call the wrapper
    obj   = lambda unknowns:get_objective(unknowns,(segment,state))   
    econ  = lambda unknowns:get_econstraints(unknowns,(segment,state)) 
    #iecon = lambda unknowns:get_ieconstraints(unknowns,(segment,state)) 
    
    bnds = make_bnds(unknowns, state)
    
    #unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ, f_ieqcons=iecon,iter=200)
    #unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ,iter=200)
    
    unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ, bounds=bnds,iter=200)
                      
    return
    
# ----------------------------------------------------------------------
#  Helper Functions
# ----------------------------------------------------------------------
    

def get_objective(unknowns,(segment,state)):
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)
        
    objective = state.objective_value
    
    #print unknowns
    
    return objective

def get_econstraints(unknowns,(segment,state)):
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)

    constraints = state.constraint_values
    
    print constraints
        
    return constraints


def make_bnds(unknowns,state):
    
    state.unknowns.throttle
    
    ones = np.ones_like(state.unknowns.throttle)
    
    throttle_bnds = ones*(0,1)
    body_angle    = ones*(0. * Units.degrees,30. * Units.degrees)
    alts          = ones*(-1e20,1e20)
    vels          = ones*(-1e20,1e20)
    times         = np.array([0,1e20])
    
    bnds = np.vstack([throttle_bnds,body_angle,alts,vels,times])
    
    bnds = list(map(tuple, bnds))
    
    return bnds


def get_ieconstraints(unknowns,(segment,state)):

    throttles = state.unknowns.throttle
    
    
    constraints_1 = throttles
    constraints_2 = 1 - throttles
    
    constraints = np.vstack([constraints_1,constraints_2])
    
    constraints = -constraints.flatten()
    
    print 'Inequality Constraints'
    
    return constraints


# optimize.py
# 
# Created:  Dec 2016, E. Botero
# Modified: Jun 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy.optimize as opt
import numpy as np

from SUAVE.Core.Arrays import array_type
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Converge Root
# ----------------------------------------------------------------------

def converge_opt(segment,state):
    
    # pack up the array
    unknowns = state.unknowns.pack_array()
    
    # Have the optimizer call the wrapper
    obj   = lambda unknowns:get_objective(unknowns,(segment,state))   
    econ  = lambda unknowns:get_econstraints(unknowns,(segment,state)) 
    iecon = lambda unknowns:get_ieconstraints(unknowns,(segment,state)) 
    
    # Setup the bnds of the problem
    bnds = make_bnds(unknowns, (segment,state))
    
    # Solve the problem
    unknowns = opt.fmin_slsqp(obj,unknowns,f_eqcons=econ,f_ieqcons=iecon,bounds=bnds,iter=2000)
    
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
    
    return objective

def get_econstraints(unknowns,(segment,state)):
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)

    constraints = state.constraint_values
    
    return constraints


def make_bnds(unknowns,(segment,state)):
    
    ones    = state.ones_row(1)
    ones_m1 = state.ones_row_m1(1)
    ones_m2 = state.ones_row_m2(1)
    
    throttle_bnds = ones*(0.,1.)
    body_angle    = ones*(0., 50.)
    gamma         = ones*(0., 50.)
    
    if segment.air_speed_end is None:
        vels      = ones_m1*(0.,2000.)
    elif segment.air_speed_end is not None:    
        vels      = ones_m2*(0.,2000.)
    
    bnds = np.vstack([vels,throttle_bnds,gamma,body_angle])
    
    bnds = list(map(tuple, bnds))
    
    return bnds


def get_ieconstraints(unknowns,(segment,state)):
    
    if isinstance(unknowns,array_type):
        state.unknowns.unpack_array(unknowns)
    else:
        state.unknowns = unknowns
        
    if not np.all(state.inputs_last == state.unknowns):       
        segment.process.iterate(segment,state)
    
    # Time goes forward, not backward
    t_final = state.conditions.frames.inertial.time[-1,0]
    constraints = (state.conditions.frames.inertial.time[1:,0] - state.conditions.frames.inertial.time[0:-1,0])/t_final
    
    # Less than a specified CL limit
    CL_limit = segment.CL_limit 
    constraints2 = (CL_limit  - state.conditions.aerodynamics.lift_coefficient[:,0])/CL_limit
    
    # Altitudes are greater than 0
    constraints3 = state.conditions.freestream.altitude[:,0]/segment.altitude_end
    
    constraints = np.concatenate((constraints,constraints2,constraints3))
    
    return constraints


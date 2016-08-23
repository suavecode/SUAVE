# Common.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    throttle   = state.unknowns.throttle
    body_angle = state.unknowns.body_angle
    
    # apply unknowns
    state.conditions.propulsion.throttle[:,0]            = throttle[:,0]
    state.conditions.frames.body.inertial_rotations[:,1] = body_angle[:,0]   
    

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector
    m  = state.conditions.weights.total_mass[:,0] 
    
    # horizontal
    state.residuals.forces[:,0] = np.sqrt( FT[:,0]**2. + FT[:,1]**2. )/m
    # vertical
    state.residuals.forces[:,1] = FT[:,2]/m

    return
    
    
 
    
    
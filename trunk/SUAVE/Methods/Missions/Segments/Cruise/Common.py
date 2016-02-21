# Common.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

def unpack_unknowns(segment,state):
    
    # unpack unknowns
    throttle = state.unknowns.throttle
    theta    = state.unknowns.body_angle
    rots     = state.conditions.frames.body.inertial_rotations
    
    # apply unknowns
    rotated = np.transpose(np.array([rots[:,0],np.transpose(theta[:,0]),rots[:,2]]))
    
    ones_row = state.ones_row
    ones = ones_row(1)
    
    state.conditions.propulsion.throttle = np.reshape(np.transpose(np.array([ones[:,0],np.transpose(throttle[:,0])]))[:,1],(len(ones),1))
    state.conditions.frames.body.inertial_rotations = rotated

# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector
    
    # horizontal
    res_1 = np.sqrt( FT[:,0]**2. + FT[:,1]**2. )
    # vertical
    res_2 = FT[:,2]
    
    state.residuals.forces = np.transpose(np.array([res_1,res_2]))

    return
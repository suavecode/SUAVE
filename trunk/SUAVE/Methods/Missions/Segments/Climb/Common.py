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
    
    state.conditions.propulsion.throttle = throttle
    state.conditions.frames.body.inertial_rotations = rotated
    
    return
    
# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

def residual_total_forces(segment,state):
    
    FT = state.conditions.frames.inertial.total_force_vector
    a  = state.conditions.frames.inertial.acceleration_vector
    m  = state.conditions.weights.total_mass    
    
    res_1 = FT[:,0]/m[:,0] - a[:,0]
    res_2 = FT[:,2]/m[:,0] - a[:,2]   
    
    state.residuals.forces = np.transpose(np.array([res_1,res_2]))
    
    return
       
def update_differentials_altitude(segment,state):
    """ Segment.update_differentials_altitude(conditions, numerics, unknowns)
        updates the differential operators t, D and I
        must return in dimensional time, with t[0] = 0

        Works with a segment discretized in vertical position, altitude

        Inputs - 
            unknowns      - data dictionary of segment free unknowns
            conditions    - data dictionary of segment conditions
            numerics - data dictionary of non-dimensional differential operators

        Outputs - 
            numerics - udpated data dictionary with dimensional numerics 

        Assumptions - 
            outputed operators are in dimensional time for the current solver iteration
            works with a segment discretized in vertical position, altitude

    """

    # unpack
    t = state.numerics.dimensionless.control_points
    D = state.numerics.dimensionless.differentiate
    I = state.numerics.dimensionless.integrate

    r = state.conditions.frames.inertial.position_vector
    v = state.conditions.frames.inertial.velocity_vector

    dz = r[-1,2] - r[0,2]
    vz = v[:,2,None] # maintain column array

    # get overall time step
    dt = np.dot( I[-1,:] * dz , 1/ vz[:,0] )

    # rescale operators
    t = t * dt

    # pack
    t_initial = state.conditions.frames.inertial.time[0,0]
    state.conditions.frames.inertial.time[:,0] = t_initial + t[:,0]

    return
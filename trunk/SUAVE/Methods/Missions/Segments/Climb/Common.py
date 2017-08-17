## @ingroup Methods-Missions-Segments-Climb
# Common.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jul 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def unpack_unknowns(segment,state):
    """Unpacks the unknowns set in the mission to be available for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns.throttle            [Unitless]
    state.unknowns.body_angle          [Radians]

    Outputs:
    state.conditions.propulsion.throttle            [Unitless]
    state.conditions.frames.body.inertial_rotations [Radians]

    Properties Used:
    N/A
    """        
    
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
    
    return
    
# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def residual_total_forces(segment,state):
    """Takes the summation of forces and makes a residual from the accelerations.

    Assumptions:
    No higher order terms.

    Source:
    N/A

    Inputs:
    state.conditions.frames.inertial.total_force_vector   [Newtons]
    sstate.conditions.frames.inertial.acceleration_vector [meter/second^2]
    state.conditions.weights.total_mass                   [kilogram]

    Outputs:
    state.residuals.forces                                [Unitless]

    Properties Used:
    N/A
    """        
    
    FT = state.conditions.frames.inertial.total_force_vector
    a  = state.conditions.frames.inertial.acceleration_vector
    m  = state.conditions.weights.total_mass    
    
    res_1 = FT[:,0]/m[:,0] - a[:,0]
    res_2 = FT[:,2]/m[:,0] - a[:,2]   
    
    state.residuals.forces = np.transpose(np.array([res_1,res_2]))
    
    return
      
## @ingroup Methods-Missions-Segments-Climb 
def update_differentials_altitude(segment,state):
    """ On each iteration creates the differentials and integration funcitons from knowns about the problem. Sets the time at each point. Must return in dimensional time, with t[0] = 0

        Assumptions:
        Works with a segment discretized in vertical position, altitude

        Inputs:
        state.numerics.dimensionless.control_points      [Unitless]
        state.numerics.dimensionless.differentiate       [Unitless]
        state.numerics.dimensionless.integrate           [Unitless]
        state.conditions.frames.inertial.position_vector [meter]
        state.conditions.frames.inertial.velocity_vector [meter/second]
        

        Outputs:
        state.conditions.frames.inertial.time            [second]


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
## @ingroup Methods-Missions-Segments-Climb
# Common.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jul 2017, E. Botero
#           Mar 2020, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Climb
def unpack_unknowns(segment):
    """Unpacks the unknowns set in the mission to be available for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment.state.unknowns.throttle            [Unitless]
    segment.state.unknowns.body_angle          [Radians]

    Outputs:
    segment.state.conditions.propulsion.throttle            [Unitless]
    segment.state.conditions.frames.body.inertial_rotations [Radians]

    Properties Used:
    N/A
    """        
    
    # unpack unknowns
    throttle = segment.state.unknowns.throttle
    theta    = segment.state.unknowns.body_angle
    
    # apply unknowns
    segment.state.conditions.propulsion.throttle[:,0]            = throttle[:,0]
    segment.state.conditions.frames.body.inertial_rotations[:,1] = theta[:,0]   
    
# ----------------------------------------------------------------------
#  Residual Total Forces
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def residual_total_forces(segment):
    """Takes the summation of forces and makes a residual from the accelerations.

    Assumptions:
    No higher order terms.

    Source:
    N/A

    Inputs:
    segment.state.conditions.frames.inertial.total_force_vector   [Newtons]
    segment.state.conditions.frames.inertial.acceleration_vector  [meter/second^2]
    segment.state.conditions.weights.total_mass                   [kilogram]

    Outputs:
    segment.state.residuals.forces                                [Unitless]

    Properties Used:
    N/A
    """        
    
    FT = segment.state.conditions.frames.inertial.total_force_vector
    a  = segment.state.conditions.frames.inertial.acceleration_vector
    m  = segment.state.conditions.weights.total_mass    
    
    segment.state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    segment.state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]       

    return
      
## @ingroup Methods-Missions-Segments-Climb 
def update_differentials_altitude(segment):
    """ On each iteration creates the differentials and integration functions from knowns about the problem. Sets the time at each point. Must return in dimensional time, with t[0] = 0

        Assumptions:
        Works with a segment discretized in vertical position, altitude

        Inputs:
        segment.state.numerics.dimensionless.control_points      [Unitless]
        segment.state.numerics.dimensionless.differentiate       [Unitless]
        segment.state.numerics.dimensionless.integrate           [Unitless]
        segment.state.conditions.frames.inertial.position_vector [meter]
        segment.state.conditions.frames.inertial.velocity_vector [meter/second]
        

        Outputs:
        segment.state.conditions.frames.inertial.time            [second]


    """

    # unpack
    t = segment.state.numerics.dimensionless.control_points 
    I = segment.state.numerics.dimensionless.integrate
    r = segment.state.conditions.frames.inertial.position_vector
    v = segment.state.conditions.frames.inertial.velocity_vector

    dz = r[-1,2] - r[0,2]
    vz = v[:,2,None] # maintain column array

    # get overall time step
    dt = np.dot( I[-1,:] * dz , 1/ vz[:,0] )

    # rescale operators
    t = t * dt

    # pack
    t_initial = segment.state.conditions.frames.inertial.time[0,0]
    segment.state.conditions.frames.inertial.time[:,0] = t_initial + t[:,0]

    return
## @ingroup Methods-Missions-Segments-Descent
# Constant_Throttle_Constant_Rate.py
#
# Created: Mar, 2018, A.A. Wachman


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np

# ----------------------------------------------------------------------
#  Unpack Unknowns
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Descent
# get initial body angle 
def unpack_body_angle(segment,state):
    """Gets the initial value for the body angle.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    state.unknowns.body_angle                      [Radians]

    Outputs:
    state.conditions.frames.body.inertial_rotation [Radians]

    Properties Used:
    None
    """

    # unpack unknowns
    theta      = state.unknowns.body_angle

    # apply unknowns
    state.conditions.frames.body.inertial_rotations[:,1] = theta[:,0]


# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Descent
# check intial altitude, and int vz to get alt instead of discret alt
def initialize_conditions(segment,state):
    """Sets the specified conditions which are given for the segment type.
    
    Uses estimated air speed instead of air speed to attempt to preserve
    descent rate.
    
    Assumptions:
    Constant throttle setting, with a constant rate of descent.
    
    Alternatively, descent_rate could be calculated as an input
    using a required L/D ratio.

    Source:
    N/A

    Inputs:
    segment.descent_rate                        [meters/second]
    segment.air_speed                           [meters/second]
    segment.throttle                            [Unitless]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]
    state.numerics.dimensionless.control_points [Unitless]
    conditions.freestream.density               [kilograms/meter^3]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.propulsion.throttle              [Unitless]

    Properties Used:
    N/A
    """  

    # unpack
    throttle     = segment.throttle  # this gets set/assumed but should not be used - set aircraft to have zero engine configuration (number of engines = 0.)
    descent_rate = segment.descent_rate
    eas          = segment.equivalent_air_speed
    alt0         = segment.altitude_start
    altf         = segment.altitude_end
    t_nondim     = state.numerics.dimensionless.control_points
    conditions   = state.conditions


    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]

    # pack conditions, these are the constants.
    conditions.propulsion.throttle[:,0] = throttle
    conditions.frames.inertial.velocity_vector[:,2] = descent_rate # positive because z is down

## @ingroup Methods-Missions-Segments-Descent
def update_differentials_altitude(segment,state):
    """On each iteration creates the differentials and integration
    functions from knowns about the problem. Sets the time at each
    point. Must return in dimensional time, with t[0] = 0.
    
    Uses estimated air speed instead of air speed to attempt to 
    preserve descent rate.
    
    Assumptions:
    Constant throttle setting, with a constant rate of descent

    Source:
    N/A

    Inputs:
    segment.descent_angle                               [radians]
    state.conditions.frames.inertial.velocity_vector    [meter/second]
    segment.altitude_start                              [meters]
    segment.altitude_end                                [meters]

    Outputs:
    state.conditions.frames.inertial.time               [seconds]
    conditions.frames.inertial.position_vector          [meters]
    conditions.freestream.altitude                      [meters]

    Properties Used:
    N/A
    """ 
   
    # unpack
    t = state.numerics.dimensionless.control_points
    D = state.numerics.dimensionless.differentiate
    I = state.numerics.dimensionless.integrate

    
    # Unpack segment initials
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end    
    conditions = state.conditions  
    v          = state.conditions.frames.inertial.velocity_vector
    desRate    = segment.descent_rate
    
    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2] 
   
    # check for incorrectly assigned descent direction
    if desRate < 0:
        raise AttributeError('descent rate set as upward (climb), check value - should be positive as z is downwards.')
        desRate = -segment.descent_rate
    
    # check of zero descent rate
    if desRate == 0:
        raise AttributeError('descent rate set as zero, invalid for mission, please set positive descent value.')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]


    # get overall time step
    vz = -v[:,2,None]   # Inertial velocity is z down, this should be pos
    vx = v[:,0,None]    # Inertial forward velocity, assumes plane is not flying backwards
    dz = altf- alt0    
    dt = dz / np.dot(I[-1,:],vz)[-1] # maintain column array
    
    # Integrate vz to get altitudes
    alt = alt0 + np.dot(I*dt,vz)

    # rescale operators
    t = t * dt

    # pack
    t_initial = state.conditions.frames.inertial.time[0,0]
    state.conditions.frames.inertial.time[:,0] = t_initial + t[:,0]
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude, not in aircraft frame 

    return

# ----------------------------------------------------------------------
#  Update Velocity Vector from Wind Angle
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Descent
def update_velocity_vector_from_wind_angle(segment,state):
    # unpack num 1
    eas = segment.equivalent_air_speed
    conditions   = state.conditions
    descent_rate = segment.descent_rate
    # determine airspeed from equivalent airspeed
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state) # get density for airspeed
    density   = conditions.freestream.density[:,0]   
    MSL_data  = segment.analyses.atmosphere.compute_values(0.0,segment.temperature_deviation)
    air_speed = eas/np.sqrt(density/MSL_data.density[0]) # convert eas to as    


    # unpack
    conditions = state.conditions 
    v_mag      = air_speed

    # process
    v_z = descent_rate # z points down, this should be positive
    v_x   = np.sqrt( v_mag**2 - v_z**2 )

    # pack
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z

    return conditions

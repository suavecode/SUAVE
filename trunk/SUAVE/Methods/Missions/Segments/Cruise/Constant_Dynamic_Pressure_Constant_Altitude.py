## @ingroup Methods-Missions-Segments-Cruise
# Constant_Dynamic_Pressure_Constant_Altitude.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
import autograd.numpy as np

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Cruise
def initialize_conditions(segment,state):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constant dynamic pressure and constant altitude

    Source:
    N/A

    Inputs:
    segment.altitude                [meters]
    segment.distance                [meters]
    segment.dynamic_pressure        [pascals]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """      
    
    # unpack
    alt        = segment.altitude
    xf         = segment.distance
    q          = segment.dynamic_pressure
    conditions = state.conditions   
    
    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho        = conditions.freestream.density[:,0]   
    
    # check for initial altitude
    if alt is None:
        if not state.initials: raise AttributeError('altitude not set')
        alt = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
        segment.altitude = alt        
    
    # compute speed, constant with constant altitude
    air_speed = np.sqrt(q/(rho*0.5))
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_final   = xf / air_speed + t_initial
    t_nondim  = state.numerics.dimensionless.control_points
    time      = t_nondim * (t_final-t_initial) + t_initial
    
    alts      = state.ones_row(1) * alt
    cond      = state.conditions
    airspeeds = np.atleast_2d(air_speed).T
    
    # pack
    cond.freestream.altitude             = alts
    cond.frames.inertial.position_vector = np.concatenate((cond.frames.inertial.position_vector[:,0:2],-alts),axis=1)
    cond.frames.inertial.velocity_vector = np.concatenate((airspeeds,cond.frames.inertial.velocity_vector[:,1:] ),axis=1)
    cond.frames.inertial.time            = time
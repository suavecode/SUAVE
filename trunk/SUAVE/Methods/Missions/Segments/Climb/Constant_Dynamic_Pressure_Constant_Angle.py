## @ingroup Methods-Missions-Segments-Climb
# Constant_Dynamic_Pressure_Constant_Angle.py
# 
# Created:  Jun 2017, E. Botero
# Modified:          

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np
import SUAVE

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Climb
def initialize_conditions_unpack_unknowns(segment,state):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constrant dynamic pressure and constant rate of climb

    Source:
    N/A

    Inputs:
    segment.climb_angle                         [radians]
    segment.dynamic_pressure                    [pascals]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]
    state.numerics.dimensionless.control_points [unitless]
    conditions.freestream.density               [kilograms/meter^3]
    state.unknowns.throttle                     [unitless]
    state.unknowns.body_angle                   [radians]
    state.unknowns.altitudes                    [meter]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.propulsion.throttle              [unitless]
    conditions.frames.body.inertial_rotations   [radians]

    Properties Used:
    N/A
    """           
    
    # unpack
    climb_angle = segment.climb_angle
    q           = segment.dynamic_pressure
    alt0        = segment.altitude_start 
    altf        = segment.altitude_end
    t_nondim    = state.numerics.dimensionless.control_points
    conditions  = state.conditions
    rho         = conditions.freestream.density[:,0]
    
    # unpack unknowns
    throttle = state.unknowns.throttle
    theta    = state.unknowns.body_angle
    alts     = state.unknowns.altitudes    

    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho = conditions.freestream.density[:,0]   

    # check for initial altitude
    if alt0 is None:
        if not state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # pack conditions    
    conditions.freestream.altitude[:,0]             =  alts[:,0] # positive altitude in this context    
    
    # Update freestream to get density
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment,state)
    rho = conditions.freestream.density[:,0]       
    
    # process velocity vector
    v_mag = np.sqrt(2*q/rho)
    v_x   = v_mag * np.cos(climb_angle)
    v_z   = -v_mag * np.sin(climb_angle)
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alts[:,0] # z points down
    conditions.propulsion.throttle[:,0]             = throttle[:,0]
    conditions.frames.body.inertial_rotations[:,1]  = theta[:,0]  
    
## @ingroup Methods-Missions-Segments-Climb
def residual_total_forces(segment,state):
    """Takes the summation of forces and makes a residual from the accelerations and altitude.

    Assumptions:
    No higher order terms.

    Source:
    N/A

    Inputs:
    state.conditions.frames.inertial.total_force_vector   [Newtons]
    sstate.conditions.frames.inertial.acceleration_vector [meter/second^2]
    state.conditions.weights.total_mass                   [kilogram]
    state.conditions.freestream.altitude                  [meter]

    Outputs:
    state.residuals.forces                                [Unitless]

    Properties Used:
    N/A
    """     
    
    # Unpack results
    FT = state.conditions.frames.inertial.total_force_vector
    a  = state.conditions.frames.inertial.acceleration_vector
    m  = state.conditions.weights.total_mass    
    alt_in  = state.unknowns.altitudes[:,0] 
    alt_out = state.conditions.freestream.altitude[:,0] 
    
    # Residual in X and Z, as well as a residual on the guess altitude
    state.residuals.forces[:,0] = FT[:,0]/m[:,0] - a[:,0]
    state.residuals.forces[:,1] = FT[:,2]/m[:,0] - a[:,2]
    state.residuals.forces[:,2] = (alt_in - alt_out)/alt_out[-1]

    return
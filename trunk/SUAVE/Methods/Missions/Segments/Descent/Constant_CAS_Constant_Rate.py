## @ingroup Methods-Missions-Segments-Descent
# Constant_CAS_Constant_Rate.py
# 
# Created:  Aug 2020, S. Karpuk
# Modified: 
#
# Adapted from Constant_CAS_Constant_Rate

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------
## @ingroup Methods-Missions-Segments-Descent
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Constant EAS speed and constant descent rate

    Source:
    N/A

    Inputs:
    segment.equivalent_air_speed                        [meters/second]
    segment.altitude_start                              [meters]
    segment.altitude_end                                [meters]
    segment.descent_rate                                [meters/second]
    segment.state.numerics.dimensionless.control_points [array]

    Outputs:
    conditions.frames.inertial.velocity_vector  [meters/second]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]
    conditions.frames.inertial.time             [seconds]

    Properties Used:
    N/A
    """       
    
    # unpack
    descent_rate = segment.descent_rate
    cas          = segment.calibrated_air_speed   
    alt0         = segment.altitude_start 
    altf         = segment.altitude_end
    t_nondim     = segment.state.numerics.dimensionless.control_points
    conditions   = segment.state.conditions  

    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 * segment.state.initials.conditions.frames.inertial.position_vector[-1,2]

    # discretize on altitude
    alt = t_nondim * (altf-alt0) + alt0
    
    # determine airspeed from calibrated airspeed
    SUAVE.Methods.Missions.Segments.Common.Aerodynamics.update_atmosphere(segment) # get density for airspeed

    alt_data = segment.analyses.atmosphere.compute_values(alt,segment.temperature_deviation)
    density  = alt_data.density[:,0]  
    pressure = alt_data.pressure[:,0] 

    MSL_data  = segment.analyses.atmosphere.compute_values(0.0,segment.temperature_deviation)
    pressure0 = MSL_data.pressure[0]

    kcas  = cas / Units.knots
    delta = pressure / pressure0 

    mach = 2.236*((((1+4.575e-7*kcas**2)**3.5-1)/delta + 1)**0.2857 - 1)**0.5

    qc  = pressure * ((1+0.2*mach**2)**3.5 - 1)
    eas = cas * (pressure/pressure0)**0.5*(((qc/pressure+1)**0.286-1)/((qc/pressure0+1)**0.286-1))**0.5
    
    air_speed = eas/np.sqrt(density/MSL_data.density[0])    
    
    # process velocity vector
    v_mag = air_speed
    v_z   = descent_rate # z points down
    v_x   = np.sqrt( v_mag**2 - v_z**2 )
    
    # pack conditions    
    conditions.frames.inertial.velocity_vector[:,0] = v_x
    conditions.frames.inertial.velocity_vector[:,2] = v_z
    conditions.frames.inertial.position_vector[:,2] = -alt[:,0] # z points down
    conditions.freestream.altitude[:,0]             =  alt[:,0] # positive altitude in this context
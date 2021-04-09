## @ingroup Methods-Missions-Segments-Ground
# Takeoff.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from . import Common

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------  

## @ingroup Methods-Missions-Segments-Ground
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Builds on the initialize conditions for common

    Source:
    N/A

    Inputs:
    segment.throttle                           [unitless]
    conditions.frames.inertial.position_vector [meters]
    conditions.weights.total_mass              [kilogram]

    Outputs:
    conditions.weights.total_mass              [kilogram]
    conditions.frames.inertial.position_vector [unitless]
    conditions.propulsion.throttle             [meters]
    
    Properties Used:
    N/A
    """  

    # use the common initialization
    Common.initialize_conditions(segment)
    conditions = segment.state.conditions    
    
    # unpack
    throttle  = segment.throttle	
    r_initial = conditions.frames.inertial.position_vector[0,:][None,:]
    m_initial = segment.analyses.weights.vehicle.mass_properties.takeoff    

    # default initial time, position, and mass
    # apply initials
    conditions.weights.total_mass[:,0]              = m_initial
    conditions.frames.inertial.position_vector[:,:] = r_initial[:,:]
    conditions.propulsion.throttle[:,0]             = throttle
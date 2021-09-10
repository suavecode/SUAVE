## @ingroup Methods-Missions-Segments-Ground
# Landing.py
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
    segment.throttle                                         [unitless]
    segment.analyses.weights.vehicle.mass_properties.landing [kilogram]
    
    Outputs:
    conditions.weights.total_mass   [kilogram]
    conditions.propulsion.throttle  [unitless]

    Properties Used:
    N/A
    """      
    
    # use the common initialization
    conditions = segment.state.conditions
    Common.initialize_conditions(segment)
    
    # Unpack
    throttle  = segment.throttle
    m_initial = segment.analyses.weights.vehicle.mass_properties.landing
          
    # apply initials
    conditions.weights.total_mass[:,0]  = m_initial
    conditions.propulsion.throttle[:,0] = throttle        

    return conditions
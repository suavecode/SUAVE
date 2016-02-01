# Landing.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import Common

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------   

def initialize_conditions(segment,state):
    
    Common.initialize_conditions(segment,state)
    
    m_initial = segment.analyses.weights.vehicle.mass_properties.landing
          
    # apply initials
    conditions = state.conditions
    conditions.weights.total_mass[:,0] = m_initial

    throttle = segment.throttle	
    conditions.propulsion.throttle[:,0] = throttle        

    return conditions
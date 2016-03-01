# Takeoff.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import Common
import numpy as np

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------  

def initialize_conditions(segment,state):

    Common.initialize_conditions(segment,state)
    conditions = state.conditions

    # default initial time, position, and mass
    r_initial = conditions.frames.inertial.position_vector[0,:][None,:]
    m_initial = segment.analyses.weights.vehicle.mass_properties.takeoff

    # apply initials
    conditions.weights.total_mass[:,0]   = m_initial
    conditions.frames.inertial.position_vector[:,:] = r_initial[:,:]

    throttle = segment.throttle	
    conditions.propulsion.throttle[:,0] = throttle
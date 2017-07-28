## @ingroup Methods-Missions-Segments-Common
# Weights.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np


# ----------------------------------------------------------------------
#  Initialize Weights
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def initialize_weights(segment,state):
 
    if state.initials:
        m_initial = state.initials.conditions.weights.total_mass[-1,0]
    else:
       
        m_initial = segment.analyses.weights.vehicle.mass_properties.takeoff

    m_current = state.conditions.weights.total_mass
    
    state.conditions.weights.total_mass[:,:] = m_current + (m_initial - m_current[0,0])
        
    return
    
# ----------------------------------------------------------------------
#  Update Gravity
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_gravity(segment,state):

    # unpack
    planet = segment.analyses.planet
    g0 = planet.features.sea_level_gravity       # m/s^2

    # calculate
    g = g0        # m/s^2 (placeholder for better g models)

    # pack
    state.conditions.freestream.gravity[:,0] = g

    return

# ----------------------------------------------------------------------
#  Update Weights
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_weights(segment,state):
    
    # unpack
    conditions = state.conditions
    m0        = conditions.weights.total_mass[0,0]
    m_empty   = segment.analyses.weights.mass_properties.operating_empty
    mdot_fuel = conditions.weights.vehicle_mass_rate
    I         = state.numerics.time.integrate
    g         = conditions.freestream.gravity

    # calculate
    m = m0 + np.dot(I, -mdot_fuel )

    # weight
    W = m*g

    # pack
    conditions.weights.total_mass[1:,0]                  = m[1:,0] # don't mess with m0
    conditions.frames.inertial.gravity_force_vector[:,2] = W[:,0]

    return
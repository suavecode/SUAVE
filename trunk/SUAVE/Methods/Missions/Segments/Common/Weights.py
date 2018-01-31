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
    """ Sets the initial weight of the vehicle at the start of the segment
    
        Assumptions:
        Only used if there is an initial condition
        
        Inputs:
            state.initials.conditions:
                weights.total_mass     [newtons]
            state.conditions:           
                weights.total_mass     [newtons]
            
        Outputs:
            state.conditions:           
                weights.total_mass     [newtons]

        Properties Used:
        N/A
                                
    """    
    
 
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
    """ Sets the gravity for each part of the mission
    
        Assumptions:
        Fixed sea level gravity, doesn't use a gravity model yet
        
        Inputs:
        segment.analyses.planet.features.sea_level_gravity [Data] 
            
        Outputs:
        state.conditions.freestream.gravity [meters/second^2]

        Properties Used:
        N/A
                                
    """      

    # unpack
    planet = segment.analyses.planet
    g0     = planet.features.sea_level_gravity
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
    
    """ Integrate tbe mass rate to update the weights throughout a segment
    
        Assumptions:
        Only the energy network/propulsion system can change the mass
        
        Inputs:
        state.conditions:
            weights.total_mass              [kilograms]
            weights.vehicle_mass_rate       [kilograms/second]
            freestream.gravity              [meters/second^2]
        segment.analyses.weights:
            mass_properties.operating_empty [kilograms]
        state.numerics.time.integrate       [array]
            
        Outputs:
        state.conditions:
            weights.total_mass                   [kilograms]
            frames.inertial.gravity_force_vector [kilograms]

        Properties Used:
        N/A
                                
    """          
    
    # unpack
    conditions = state.conditions
    m0         = conditions.weights.total_mass[0,0]
    mdot_fuel  = conditions.weights.vehicle_mass_rate
    g          = conditions.freestream.gravity
    I          = state.numerics.time.integrate

    # calculate
    m = m0 + np.dot(I, -mdot_fuel )

    # weight
    W = m*g

    # pack
    conditions.weights.total_mass[1:,0]                  = m[1:,0] # don't mess with m0
    conditions.frames.inertial.gravity_force_vector[:,2] = W[:,0]

    return
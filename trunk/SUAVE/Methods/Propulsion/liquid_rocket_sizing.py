## @ingroup Methods-Propulsion
# liquid_rocket_sizing.py
# 
# Created:  Feb 2018, W. Maier 
# Modified: 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#   Sizing
# ----------------------------------------------------------------------
## @ingroup Methods-Propulsion
def liquid_rocket_sizing(liquid_rocket, altitude = None, delta_isa = 0, conditions = None):  
    """ This function sizes a liquid_rocket for the input design conditions.
    """    
    
    #Unpack components
    #check if altitude is passed or conditions is passed
    if(conditions):
        #use conditions
        pass
        
    else:
        #check if mach number and temperature are passed
        if( altitude==None):
            
            #raise an error
            raise NameError('The sizing conditions require an altitude')
        
        else:
            # call the atmospheric model to get the conditions at the specified altitude
            # will need update for use on other celestial bodies!
            atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
            atmo_data  = atmosphere.compute_values(altitude,delta_isa,True)          
            planet     = SUAVE.Attributes.Planets.Earth()
            
            p   = atmo_data.pressure          
            T   = atmo_data.temperature       
                
            # setup conditions
            conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()   
            conditions.propulsion = Data()
            
            # freestream conditions    
            conditions.freestream.altitude    = np.atleast_1d(altitude)
            conditions.freestream.pressure    = np.atleast_1d(p)
            conditions.freestream.temperature = np.atleast_1d(T)
            conditions.freestream.gravity     = np.atleast_1d(planet.compute_gravity(altitude))
            
            # propulsion conditions
            conditions.propulsion.throttle    = np.atleast_1d(1.0)
    
    combustor                 = liquid_rocket.combustor
    core_nozzle               = liquid_rocket.core_nozzle
    thrust                    = liquid_rocket.thrust
    number_of_engines         = liquid_rocket.number_of_engines
    
    #--creating the network by manually linking the different components--
    
    # flow through the high pressor comprresor
    combustor.compute(conditions)
    
    # link the core nozzle to the combustor
    core_nozzle.inputs = combustor.outputs
    
    # flow through the core nozzle
    core_nozzle.compute(conditions)

    # link the thrust component to the core nozzle
    thrust.inputs                   = core_nozzle.outputs
    thrust.inputs.number_of_engines = number_of_engines
        
    # compute the thrust 
    thrust.size(conditions)
    
    #update the design thrust value
    liquid_rocket.design_thrust = thrust.total_design

    #--Compute the sls_thrust--
    
    # call the atmospheric model to get the conditions at the specified altitude
    atmosphere_sls  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data       = atmosphere_sls.compute_values(0.0,0.0)
    
    p   = atmo_data.pressure          
    T   = atmo_data.temperature       
    
    # setup conditions
    conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()   
    conditions_sls.propulsion = Data()

    # freestream conditions    
    conditions_sls.freestream.altitude    = np.atleast_1d(0.)
    conditions_sls.freestream.pressure    = np.atleast_1d(p)
    conditions_sls.freestream.temperature = np.atleast_1d(T)
    conditions_sls.freestream.gravity     = np.atleast_1d(planet.sea_level_gravity) 
    
    # propulsion conditions
    conditions_sls.propulsion.throttle    =  np.atleast_1d(1.0)    
    
    state_sls                             = Data()
    state_sls.numerics                    = Data()
    state_sls.conditions                  = conditions_sls
    results_sls                           = liquid_rocket.evaluate_thrust(state_sls)
    liquid_rocket.sealevel_static_thrust  = results_sls.thrust_force_vector[0,0] / number_of_engines
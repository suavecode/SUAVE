## @ingroup Methods-Propulsion
# scramjet_sizing.py
# 
# Created:  April 2018, W. Maier
# Modified:        
#           

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
def scramjet_sizing(scramjet,mach_number = None, altitude = None, delta_isa = 0, conditions = None):  
    """ This function sizes a scramjet for the input design conditions.
    """    
    
    #Unpack components
    #check if altitude is passed or conditions is passed
    if(conditions):
        #use conditions
        pass
        
    else:
        #check if mach number and temperature are passed
        if(mach_number==None or altitude==None):
            
            #raise an error
            raise NameError('The sizing conditions require an altitude and a Mach number')
        
        else:
            #call the atmospheric model to get the conditions at the specified altitude
            atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
            atmo_data = atmosphere.compute_values(altitude,delta_isa,True)          
            planet     = SUAVE.Attributes.Planets.Earth()
            
            p   = atmo_data.pressure          
            T   = atmo_data.temperature       
            rho = atmo_data.density          
            a   = atmo_data.speed_of_sound    
            mu  = atmo_data.dynamic_viscosity   
        
            # setup conditions
            conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            
        
            # freestream conditions    
            conditions.freestream.altitude                    = np.atleast_1d(altitude)
            conditions.freestream.mach_number                 = np.atleast_1d(mach_number)
            conditions.freestream.pressure                    = np.atleast_1d(p)
            conditions.freestream.temperature                 = np.atleast_1d(T)
            conditions.freestream.density                     = np.atleast_1d(rho)
            conditions.freestream.dynamic_viscosity           = np.atleast_1d(mu)
            conditions.freestream.gravity                     = np.atleast_1d(planet.compute_gravity(altitude))
            conditions.freestream.isentropic_expansion_factor = np.atleast_1d(scramjet.working_fluid.compute_gamma(T,p))
            conditions.freestream.Cp                          = np.atleast_1d(scramjet.working_fluid.compute_cp(T,p))
            conditions.freestream.R                           = np.atleast_1d(scramjet.working_fluid.gas_specific_constant)
            conditions.freestream.speed_of_sound              = np.atleast_1d(a)
            conditions.freestream.velocity                    = np.atleast_1d(a*mach_number)
            
            # propulsion conditions
            conditions.propulsion.throttle           =  np.atleast_1d(1.0)
    
    ram                       = scramjet.ram
    inlet_nozzle              = scramjet.inlet_nozzle
    combustor                 = scramjet.combustor
    core_nozzle               = scramjet.core_nozzle
    thrust                    = scramjet.thrust
    number_of_engines         = scramjet.number_of_engines
    
    #Creating the network by manually linking the different components
    #set the working fluid to determine the fluid properties
    ram.inputs.working_fluid       = scramjet.working_fluid
    
    #Flow through the ram
    ram(conditions)
    
    #link inlet nozzle to ram 
    inlet_nozzle.inputs             = ram.outputs
    
    #Flow through the inlet nozzle
    inlet_nozzle.compute_scramjet(conditions)

    #link the combustor to the inlet nozzle
    combustor.inputs.stagnation_temperature                = inlet_nozzle.outputs.stagnation_temperature 
    combustor.inputs.stagnation_pressure                   = inlet_nozzle.outputs.stagnation_pressure 
    combustor.inputs.inlet_nozzle                          = inlet_nozzle.outputs
        
    #flow through the high pressor comprresor
    combustor.compute_supersonic_combustion(conditions)
    
    #link the core nozzle to the combustor
    core_nozzle.inputs              = combustor.outputs
    
    #flow through the core nozzle
    core_nozzle.compute_scramjet(conditions)

    #link the thrust component to the core nozzle
    thrust.inputs.core_nozzle                              = core_nozzle.outputs
    thrust.inputs.number_of_engines                        = number_of_engines
    thrust.inputs.total_temperature_reference              = core_nozzle.outputs.stagnation_temperature
    thrust.inputs.total_pressure_reference                 = core_nozzle.outputs.stagnation_pressure
    
    #link the thrust component to the combustor
    thrust.inputs.fuel_to_air_ratio = combustor.outputs.fuel_to_air_ratio
    
    #compute the thrust
    thrust.inputs.fan_nozzle                               = Data()
    thrust.inputs.fan_nozzle.velocity                      = 0.0
    thrust.inputs.fan_nozzle.area_ratio                    = 0.0
    thrust.inputs.fan_nozzle.static_pressure               = 0.0
    thrust.inputs.bypass_ratio                             = 0.0
    thrust.inputs.flow_through_core                        = 1.0 #scaled constant to turn on core thrust computation
    thrust.inputs.flow_through_fan                         = 0.0 #scaled constant to turn on fan thrust computation     
    thrust.size_stream_thrust(conditions)
    
    #update the design thrust value
    scramjet.design_thrust = thrust.total_design

    #compute the sls_thrust
    #call the atmospheric model to get the conditions at the specified altitude
    atmosphere_sls  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data       = atmosphere_sls.compute_values(0.0,0.0)
    
    p   = atmo_data.pressure          
    T   = atmo_data.temperature       
    rho = atmo_data.density          
    a   = atmo_data.speed_of_sound    
    mu  = atmo_data.dynamic_viscosity   

    # setup conditions
    conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            

    # freestream conditions    
    conditions_sls.freestream.altitude                    = np.atleast_1d(0.)
    conditions_sls.freestream.mach_number                 = np.atleast_1d(0.01)
    conditions_sls.freestream.pressure                    = np.atleast_1d(p)
    conditions_sls.freestream.temperature                 = np.atleast_1d(T)
    conditions_sls.freestream.density                     = np.atleast_1d(rho)
    conditions_sls.freestream.dynamic_viscosity           = np.atleast_1d(mu)
    conditions_sls.freestream.gravity                     = np.atleast_1d(planet.sea_level_gravity)
    conditions_sls.freestream.isentropic_expansion_factor = np.atleast_1d(scramjet.working_fluid.compute_gamma(T,p))
    conditions_sls.freestream.Cp                          = np.atleast_1d(scramjet.working_fluid.compute_cp(T,p))
    conditions_sls.freestream.R                           = np.atleast_1d(scramjet.working_fluid.gas_specific_constant)
    conditions_sls.freestream.speed_of_sound              = np.atleast_1d(a)
    conditions_sls.freestream.velocity                    = np.atleast_1d(a*0.01)
    
    # propulsion conditions
    conditions_sls.propulsion.throttle           =  np.atleast_1d(1.0)    
    
    state_sls            = Data()
    state_sls.numerics   = Data()
    state_sls.conditions = conditions_sls   
    results_sls          = scramjet.evaluate_thrust(state_sls)
    scramjet.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / number_of_engines
# turbofan_sizing.py
# 
# Created:  Mar 2015, A. Variyar 
# Modified: Feb 2016, M. Vegh
#           Jan 2016, E. Botero
#           Jan 2020, T. MacDonald

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#   Sizing
# ----------------------------------------------------------------------

def turbofan_sizing(turbofan,mach_number = None, altitude = None, delta_isa = 0, conditions = None):  
    """ create and evaluate a gas turbine network
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
            atmo_data  = atmosphere.compute_values(altitude,delta_isa)
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
            conditions.freestream.isentropic_expansion_factor = np.atleast_1d(turbofan.working_fluid.compute_gamma(T,p))
            conditions.freestream.Cp                          = np.atleast_1d(turbofan.working_fluid.compute_cp(T,p))
            conditions.freestream.R                           = np.atleast_1d(turbofan.working_fluid.gas_specific_constant)
            conditions.freestream.speed_of_sound              = np.atleast_1d(a)
            conditions.freestream.velocity                    = np.atleast_1d(a*mach_number)
            
            # propulsion conditions
            conditions.propulsion.throttle                    =  np.atleast_1d(1.0)
    
    ram                       = turbofan.ram
    inlet_nozzle              = turbofan.inlet_nozzle
    low_pressure_compressor   = turbofan.low_pressure_compressor
    high_pressure_compressor  = turbofan.high_pressure_compressor
    fan                       = turbofan.fan
    combustor                 = turbofan.combustor
    high_pressure_turbine     = turbofan.high_pressure_turbine
    low_pressure_turbine      = turbofan.low_pressure_turbine
    core_nozzle               = turbofan.core_nozzle
    fan_nozzle                = turbofan.fan_nozzle
    thrust                    = turbofan.thrust
    bypass_ratio              = turbofan.bypass_ratio
    number_of_engines         = turbofan.number_of_engines
    
    #Creating the network by manually linking the different components
    
    #set the working fluid to determine the fluid properties
    ram.inputs.working_fluid                             = turbofan.working_fluid
    
    #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
    ram(conditions)
    
    #link inlet nozzle to ram 
    inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature
    inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure
    #Flow through the inlet nozzle
    inlet_nozzle(conditions)      
                    
    #--link low pressure compressor to the inlet nozzle
    low_pressure_compressor.inputs.stagnation_temperature  = inlet_nozzle.outputs.stagnation_temperature
    low_pressure_compressor.inputs.stagnation_pressure     = inlet_nozzle.outputs.stagnation_pressure
    
    #Flow through the low pressure compressor
    low_pressure_compressor(conditions)
    
    #link the high pressure compressor to the low pressure compressor
    high_pressure_compressor.inputs.stagnation_temperature = low_pressure_compressor.outputs.stagnation_temperature
    high_pressure_compressor.inputs.stagnation_pressure    = low_pressure_compressor.outputs.stagnation_pressure
    
    #Flow through the high pressure compressor
    high_pressure_compressor(conditions)
    
    #Link the fan to the inlet nozzle
    fan.inputs.stagnation_temperature                      = inlet_nozzle.outputs.stagnation_temperature
    fan.inputs.stagnation_pressure                         = inlet_nozzle.outputs.stagnation_pressure
    
    #flow through the fan
    fan(conditions)
    
    #link the combustor to the high pressure compressor
    combustor.inputs.stagnation_temperature                = high_pressure_compressor.outputs.stagnation_temperature
    combustor.inputs.stagnation_pressure                   = high_pressure_compressor.outputs.stagnation_pressure
    
    #flow through the high pressor comprresor
    combustor(conditions)
    
    #link the high pressure turbione to the combustor
    high_pressure_turbine.inputs.stagnation_temperature    = combustor.outputs.stagnation_temperature
    high_pressure_turbine.inputs.stagnation_pressure       = combustor.outputs.stagnation_pressure
    high_pressure_turbine.inputs.fuel_to_air_ratio         = combustor.outputs.fuel_to_air_ratio
    
    #link the high pressure turbine to the high pressure compressor
    high_pressure_turbine.inputs.compressor                = high_pressure_compressor.outputs
    
    #link the high pressure turbine to the fan
    high_pressure_turbine.inputs.fan                       = fan.outputs
    high_pressure_turbine.inputs.bypass_ratio              = 0.0 #set to zero to ensure that fan not linked here
    
    #flow through the high pressure turbine
    high_pressure_turbine(conditions)
            
    #link the low pressure turbine to the high pressure turbine
    low_pressure_turbine.inputs.stagnation_temperature     = high_pressure_turbine.outputs.stagnation_temperature
    low_pressure_turbine.inputs.stagnation_pressure        = high_pressure_turbine.outputs.stagnation_pressure
    
    #link the low pressure turbine to the low_pressure_compresor
    low_pressure_turbine.inputs.compressor                 = low_pressure_compressor.outputs
    
    #link the low pressure turbine to the combustor
    low_pressure_turbine.inputs.fuel_to_air_ratio          = combustor.outputs.fuel_to_air_ratio
    
    #link the low pressure turbine to the fan
    low_pressure_turbine.inputs.fan                        =  fan.outputs
    
    #get the bypass ratio from the thrust component
    low_pressure_turbine.inputs.bypass_ratio               =  bypass_ratio
    
    #flow through the low pressure turbine
    low_pressure_turbine(conditions)
    
    #link the core nozzle to the low pressure turbine
    core_nozzle.inputs.stagnation_temperature              = low_pressure_turbine.outputs.stagnation_temperature
    core_nozzle.inputs.stagnation_pressure                 = low_pressure_turbine.outputs.stagnation_pressure
    
    #flow through the core nozzle
    core_nozzle(conditions)
   
    #link the fan nozzle to the fan
    fan_nozzle.inputs.stagnation_temperature               = fan.outputs.stagnation_temperature
    fan_nozzle.inputs.stagnation_pressure                  = fan.outputs.stagnation_pressure
    
    #flow through the fan nozzle
    fan_nozzle(conditions)
    
    # compute the thrust using the thrust component
    #link the thrust component to the fan nozzle
    thrust.inputs.fan_exit_velocity                        = fan_nozzle.outputs.velocity
    thrust.inputs.fan_area_ratio                           = fan_nozzle.outputs.area_ratio
    thrust.inputs.fan_nozzle                               = fan_nozzle.outputs
    
    #link the thrust component to the core nozzle
    thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
    thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
    thrust.inputs.core_nozzle                              = core_nozzle.outputs
    
    #link the thrust component to the combustor
    thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio
    
    #link the thrust component to the low pressure compressor 
    thrust.inputs.total_temperature_reference              = low_pressure_compressor.outputs.stagnation_temperature
    thrust.inputs.total_pressure_reference                 = low_pressure_compressor.outputs.stagnation_pressure
    thrust.inputs.number_of_engines                        = number_of_engines
    thrust.inputs.bypass_ratio                             = bypass_ratio
    thrust.inputs.flow_through_core                        =  1./(1.+bypass_ratio) #scaled constant to turn on core thrust computation
    thrust.inputs.flow_through_fan                         =  bypass_ratio/(1.+bypass_ratio) #scaled constant to turn on fan thrust computation     

    #compute the thrust
    thrust.size(conditions)
    
    #determine geometry; 
    mass_flow         = thrust.mass_flow_rate_design
    mass_flow_fan     = mass_flow*bypass_ratio
    
 
 
    U0                = conditions.freestream.velocity
    gamma             = ram.outputs.isentropic_expansion_factor
    R                 = ram.outputs.gas_specific_constant
    rho0              = conditions.freestream.density
    
    rho5_fan          = fan_nozzle.outputs.density
    U5_fan            = fan_nozzle.outputs.velocity
    
    rho5_core         = core_nozzle.outputs.density
    U5_core           = core_nozzle.outputs.velocity
    
    
    
  
    
    #update the design thrust value
    turbofan.design_thrust = thrust.total_design
    
    #compute the sls_thrust
    #call the atmospheric model to get the conditions at the specified altitude
    atmosphere_sls = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmosphere_sls.compute_values(0.0,0.0)
    
    p   = atmo_data.pressure          
    T   = atmo_data.temperature       
    rho = atmo_data.density          
    a   = atmo_data.speed_of_sound    
    mu  = atmo_data.dynamic_viscosity      

    # setup conditions
    conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            

    # freestream conditions    
    conditions_sls.freestream.altitude                    = np.atleast_2d(0.)
    conditions_sls.freestream.mach_number                 = np.atleast_2d(0.01)
    conditions_sls.freestream.pressure                    = np.atleast_2d(p)
    conditions_sls.freestream.temperature                 = np.atleast_2d(T)
    conditions_sls.freestream.density                     = np.atleast_2d(rho)
    conditions_sls.freestream.dynamic_viscosity           = np.atleast_2d(mu)
    conditions_sls.freestream.gravity                     = np.atleast_2d(planet.sea_level_gravity)
    conditions_sls.freestream.isentropic_expansion_factor = np.atleast_2d(turbofan.working_fluid.compute_gamma(T,p))
    conditions_sls.freestream.Cp                          = np.atleast_2d(turbofan.working_fluid.compute_cp(T,p))
    conditions_sls.freestream.R                           = np.atleast_2d(turbofan.working_fluid.gas_specific_constant)
    conditions_sls.freestream.speed_of_sound              = np.atleast_2d(a)
    conditions_sls.freestream.velocity                    = np.atleast_2d(a*0.01)
    
    # propulsion conditions
    conditions_sls.propulsion.throttle           =  np.atleast_2d(1.0)    
    
    #size the turbofan

    state_sls            = Data()
    state_sls.numerics   = Data()
    state_sls.conditions = conditions_sls   
    results_sls          = turbofan.evaluate_thrust(state_sls)
    
    turbofan.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / number_of_engines
  
 
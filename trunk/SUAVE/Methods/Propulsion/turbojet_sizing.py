# turbojet_sizing.py
# 
# Created:  May 2015, T. MacDonald 
# Modified: Jan 2016, E. Botero    
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

def turbojet_sizing(turbojet,mach_number = None, altitude = None, delta_isa = 0, conditions = None):  
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
            conditions.freestream.isentropic_expansion_factor = np.atleast_1d(turbojet.working_fluid.compute_gamma(T,p))
            conditions.freestream.Cp                          = np.atleast_1d(turbojet.working_fluid.compute_cp(T,p))
            conditions.freestream.R                           = np.atleast_1d(turbojet.working_fluid.gas_specific_constant)
            conditions.freestream.speed_of_sound              = np.atleast_1d(a)
            conditions.freestream.velocity                    = np.atleast_1d(a*mach_number)
            
            # propulsion conditions
            conditions.propulsion.throttle                    = np.atleast_1d(1.0)
    
    ram                       = turbojet.ram
    inlet_nozzle              = turbojet.inlet_nozzle
    low_pressure_compressor   = turbojet.low_pressure_compressor
    high_pressure_compressor  = turbojet.high_pressure_compressor
    combustor                 = turbojet.combustor
    high_pressure_turbine     = turbojet.high_pressure_turbine
    low_pressure_turbine      = turbojet.low_pressure_turbine
    core_nozzle               = turbojet.core_nozzle
    thrust                    = turbojet.thrust
    number_of_engines         = turbojet.number_of_engines
    
    #Creating the network by manually linking the different components
    #set the working fluid to determine the fluid properties
    ram.inputs.working_fluid                             = turbojet.working_fluid
    
    #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
    ram(conditions)
    
    #link inlet nozzle to ram 
    inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature #conditions.freestream.stagnation_temperature
    inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure #conditions.freestream.stagnation_pressure
    
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

    #link the combustor to the high pressure compressor
    combustor.inputs.stagnation_temperature                = high_pressure_compressor.outputs.stagnation_temperature
    combustor.inputs.stagnation_pressure                   = high_pressure_compressor.outputs.stagnation_pressure
    
    #flow through the high pressor comprresor
    combustor(conditions)

    #link the high pressure turbione to the combustor
    high_pressure_turbine.inputs.stagnation_temperature    = combustor.outputs.stagnation_temperature
    high_pressure_turbine.inputs.stagnation_pressure       = combustor.outputs.stagnation_pressure
    high_pressure_turbine.inputs.fuel_to_air_ratio         = combustor.outputs.fuel_to_air_ratio
    
    #link the high pressuer turbine to the high pressure compressor
    high_pressure_turbine.inputs.compressor                = high_pressure_compressor.outputs

    #flow through the high pressure turbine
    high_pressure_turbine.inputs.bypass_ratio = 0.0
    high_pressure_turbine.inputs.fan = Data()
    high_pressure_turbine.inputs.fan.work_done = 0.0
    high_pressure_turbine(conditions)
            
    #link the low pressure turbine to the high pressure turbine
    low_pressure_turbine.inputs.stagnation_temperature     = high_pressure_turbine.outputs.stagnation_temperature
    low_pressure_turbine.inputs.stagnation_pressure        = high_pressure_turbine.outputs.stagnation_pressure
    
    #link the low pressure turbine to the low_pressure_compresor
    low_pressure_turbine.inputs.compressor                 = low_pressure_compressor.outputs
    
    #link the low pressure turbine to the combustor
    low_pressure_turbine.inputs.fuel_to_air_ratio          = combustor.outputs.fuel_to_air_ratio
    
    #flow through the low pressure turbine
    low_pressure_turbine.inputs.bypass_ratio = 0.0
    low_pressure_turbine.inputs.fan = Data()
    low_pressure_turbine.inputs.fan.work_done = 0.0    
    low_pressure_turbine(conditions)
    
    #link the core nozzle to the low pressure turbine
    core_nozzle.inputs.stagnation_temperature              = low_pressure_turbine.outputs.stagnation_temperature
    core_nozzle.inputs.stagnation_pressure                 = low_pressure_turbine.outputs.stagnation_pressure
    
    #flow through the core nozzle
    core_nozzle(conditions)

    # compute the thrust using the thrust component
    #link the thrust component to the core nozzle
    thrust.inputs.core_exit_velocity                       = core_nozzle.outputs.velocity
    thrust.inputs.core_area_ratio                          = core_nozzle.outputs.area_ratio
    thrust.inputs.core_nozzle                              = core_nozzle.outputs
    
    #link the thrust component to the combustor
    thrust.inputs.fuel_to_air_ratio                        = combustor.outputs.fuel_to_air_ratio
    
    #link the thrust component to the low pressure compressor 
    thrust.inputs.stag_temp_lpt_exit                       = low_pressure_compressor.outputs.stagnation_temperature
    thrust.inputs.stag_press_lpt_exit                      = low_pressure_compressor.outputs.stagnation_pressure
    thrust.inputs.number_of_engines                        = number_of_engines
    thrust.inputs.total_temperature_reference              = low_pressure_compressor.outputs.stagnation_temperature
    thrust.inputs.total_pressure_reference                 = low_pressure_compressor.outputs.stagnation_pressure

    #compute the thrust
    thrust.inputs.fan_nozzle = Data()
    thrust.inputs.fan_nozzle.velocity = 0.0
    thrust.inputs.fan_nozzle.area_ratio = 0.0
    thrust.inputs.fan_nozzle.static_pressure = 0.0
    thrust.inputs.bypass_ratio = 0.0
    thrust.inputs.flow_through_core                        =  1.0 #scaled constant to turn on core thrust computation
    thrust.inputs.flow_through_fan                         =  0.0 #scaled constant to turn on fan thrust computation     
    thrust.size(conditions)
    
    #update the design thrust value
    turbojet.design_thrust = thrust.total_design

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
    conditions_sls.freestream.isentropic_expansion_factor = np.atleast_2d(turbojet.working_fluid.compute_gamma(T,p))
    conditions_sls.freestream.Cp                          = np.atleast_2d(turbojet.working_fluid.compute_cp(T,p))
    conditions_sls.freestream.R                           = np.atleast_2d(turbojet.working_fluid.gas_specific_constant)
    conditions_sls.freestream.speed_of_sound              = np.atleast_2d(a)
    conditions_sls.freestream.velocity                    = np.atleast_2d(a*0.01)
    
    # propulsion conditions
    conditions_sls.propulsion.throttle           =  np.atleast_2d(1.0)    
    
    state_sls                       = Data()
    state_sls.numerics              = Data()
    state_sls.conditions            = conditions_sls   
    results_sls                     = turbojet.evaluate_thrust(state_sls)
    turbojet.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / number_of_engines
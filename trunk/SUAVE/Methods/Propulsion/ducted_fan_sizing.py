# test_gasturbine_network.py
# 
# Created:  Michael Vegh, July 2015
# Modified: 
#        

""" create and evaluate a Ducted Fan network
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
import numpy as np
from SUAVE.Core import Data


def ducted_fan_sizing(ducted_fan,mach_number = None, altitude = None, delta_isa = 0, conditions = None):  
    '''create and evaluate a ducted_fan network
    '''
    
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
            atmo_data = atmosphere.compute_values(altitude,delta_isa)

            p   = atmo_data.pressure          
            T   = atmo_data.temperature       
            rho = atmo_data.density          
            a   = atmo_data.speed_of_sound    
            mu  = atmo_data.dynamic_viscosity   
        
            # setup conditions
            conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()            
        
        
        
            # freestream conditions
            
            conditions.freestream.altitude           = np.atleast_1d(altitude)
            conditions.freestream.mach_number        = np.atleast_1d(mach_number)
            
            conditions.freestream.pressure           = np.atleast_1d(p)
            conditions.freestream.temperature        = np.atleast_1d(T)
            conditions.freestream.density            = np.atleast_1d(rho)
            conditions.freestream.dynamic_viscosity  = np.atleast_1d(mu)
            conditions.freestream.gravity            = np.atleast_1d(9.81)
            conditions.freestream.gamma              = np.atleast_1d(1.4)
            conditions.freestream.Cp                 = 1.4*287.87/(1.4-1)
            conditions.freestream.R                  = 287.87
            conditions.freestream.speed_of_sound     = np.atleast_1d(a)
            conditions.freestream.velocity           = conditions.freestream.mach_number * conditions.freestream.speed_of_sound
            
            # propulsion conditions
            conditions.propulsion.throttle           =  np.atleast_1d(1.0)
    
    
    
    ram                       = ducted_fan.ram
    inlet_nozzle              = ducted_fan.inlet_nozzle
    fan                       = ducted_fan.fan
    fan_nozzle                = ducted_fan.fan_nozzle
    thrust                    = ducted_fan.thrust
    
    bypass_ratio              = ducted_fan.bypass_ratio #0
    number_of_engines         = ducted_fan.number_of_engines
    
    #Creating the network by manually linking the different components
    
    
    #set the working fluid to determine the fluid properties
    ram.inputs.working_fluid                             = ducted_fan.working_fluid
    
    #Flow through the ram , this computes the necessary flow quantities and stores it into conditions
    ram(conditions)
    
    
    
    #link inlet nozzle to ram 
    inlet_nozzle.inputs.stagnation_temperature             = ram.outputs.stagnation_temperature #conditions.freestream.stagnation_temperature
    inlet_nozzle.inputs.stagnation_pressure                = ram.outputs.stagnation_pressure #conditions.freestream.stagnation_pressure
    
    #Flow through the inlet nozzle
    inlet_nozzle(conditions)
        
    #Link the fan to the inlet nozzle
    fan.inputs.stagnation_temperature                      = inlet_nozzle.outputs.stagnation_temperature
    fan.inputs.stagnation_pressure                         = inlet_nozzle.outputs.stagnation_pressure
    
    #flow through the fan
    fan(conditions)
    
    
           
    

    #link the dan nozzle to the fan
    fan_nozzle.inputs.stagnation_temperature               = fan.outputs.stagnation_temperature
    fan_nozzle.inputs.stagnation_pressure                  = fan.outputs.stagnation_pressure
    
     # flow through the fan nozzle
    fan_nozzle(conditions)
    
    # compute the thrust using the thrust component
    
    
    
    #link the thrust component to the fan nozzle
    thrust.inputs.fan_exit_velocity                        = fan_nozzle.outputs.velocity
    thrust.inputs.fan_area_ratio                           = fan_nozzle.outputs.area_ratio
    thrust.inputs.fan_nozzle                               = fan_nozzle.outputs
    thrust.inputs.number_of_engines                        = number_of_engines
    thrust.inputs.bypass_ratio                             = bypass_ratio
    thrust.inputs.total_temperature_reference              = fan_nozzle.outputs.stagnation_temperature
    thrust.inputs.total_pressure_reference                 = fan_nozzle.outputs.stagnation_pressure
    thrust.inputs.flow_through_core                        = 0.
    thrust.inputs.flow_through_fan                         = 1.
    
    #nonexistant components used to run thrust
    thrust.inputs.core_exit_velocity                       = 0.
    thrust.inputs.core_area_ratio                          = 0.
    thrust.inputs.core_nozzle                              = Data()
    thrust.inputs.core_nozzle.velocity                     = 0.
    thrust.inputs.core_nozzle.area_ratio                   = 0.
    thrust.inputs.core_nozzle.static_pressure              = 0.                                                                                                                
    
    
    #compute the trust
    thrust.size(conditions)
    mass_flow  = thrust.mass_flow_rate_design
    

    #update the design thrust value
    ducted_fan.design_thrust = thrust.total_design
    
    
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
    
    conditions_sls.freestream.altitude           = np.atleast_1d(0.)
    conditions_sls.freestream.mach_number        = np.atleast_1d(0.01)
    
    conditions_sls.freestream.pressure           = np.atleast_1d(p)
    conditions_sls.freestream.temperature        = np.atleast_1d(T)
    conditions_sls.freestream.density            = np.atleast_1d(rho)
    conditions_sls.freestream.dynamic_viscosity  = np.atleast_1d(mu)
    conditions_sls.freestream.gravity            = np.atleast_1d(9.81)
    conditions_sls.freestream.gamma              = np.atleast_1d(1.4)
    conditions_sls.freestream.Cp                 = 1.4*287.87/(1.4-1)
    conditions_sls.freestream.R                  = 287.87
    conditions_sls.freestream.speed_of_sound     = np.atleast_1d(a)
    conditions_sls.freestream.velocity           = conditions_sls.freestream.mach_number * conditions_sls.freestream.speed_of_sound
    
    # propulsion conditions
    conditions_sls.propulsion.throttle           =  np.atleast_1d(1.0)    
    
    state_sls = Data()
    state_sls.numerics = Data()
    state_sls.conditions = conditions_sls   
    results_sls = ducted_fan.evaluate_thrust(state_sls)
    
    ducted_fan.sealevel_static_thrust = results_sls.thrust_force_vector[0,0] / number_of_engines
   
    
    #determine geometry
    
    
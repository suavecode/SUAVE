# ducted_fan.py
# 
# Created:  Jan 2018, W. Maier
# Modified:       

""" create and evaluate a ducted_fan network
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Core import (
Data, Container,
)

from SUAVE.Components import Component, Physical_Component, Lofted_Body
from SUAVE.Components.Energy.Networks.Ducted_Fan import Ducted_Fan
from SUAVE.Methods.Propulsion.ducted_fan_sizing import ducted_fan_sizing

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():   
    # call the network function
    energy_network()
    
    return


def energy_network():
    # ------------------------------------------------------------------
    #   Evaluation Conditions
    # ------------------------------------------------------------------    
    
    # Setup Conditions        
    ones_1col  = np.ones([1,1])    
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
       
    # Freestream conditions
    conditions.freestream.mach_number                 = ones_1col*0.4
    conditions.freestream.pressure                    = ones_1col*20000.0
    conditions.freestream.temperature                 = ones_1col*215.0
    conditions.freestream.density                     = ones_1col*0.8
    conditions.freestream.dynamic_viscosity           = ones_1col*0.000001475
    conditions.freestream.altitude                    = ones_1col*10.0
    conditions.freestream.gravity                     = ones_1col*9.81
    conditions.freestream.isentropic_expansion_factor = ones_1col*1.4
    conditions.freestream.Cp                          = 1.4*287.87/(1.4-1)
    conditions.freestream.R                           = 287.87
    conditions.M                                      = conditions.freestream.mach_number 
    conditions.T                                      = conditions.freestream.temperature
    conditions.p                                      = conditions.freestream.pressure
    conditions.freestream.speed_of_sound              = ones_1col* np.sqrt(conditions.freestream.Cp/(conditions.freestream.Cp-conditions.freestream.R)*conditions.freestream.R*conditions.freestream.temperature) #300.
    conditions.freestream.velocity                    = conditions.M * conditions.freestream.speed_of_sound
    conditions.velocity                               = conditions.M * conditions.freestream.speed_of_sound
    conditions.q                                      = 0.5*conditions.freestream.density*conditions.velocity**2
    conditions.g0                                     = conditions.freestream.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle                    =  ones_1col*1.0

    # ------------------------------------------------------------------
    #   Design/sizing conditions 
    # ------------------------------------------------------------------    
        
    # Setup Conditions        
    ones_1col = np.ones([1,1])       
    conditions_sizing = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
 
    # freestream conditions
    conditions_sizing.freestream.mach_number                 = ones_1col*0.5
    conditions_sizing.freestream.pressure                    = ones_1col*26499.73156529
    conditions_sizing.freestream.temperature                 = ones_1col*223.25186491
    conditions_sizing.freestream.density                     = ones_1col*0.41350854
    conditions_sizing.freestream.dynamic_viscosity           = ones_1col* 1.45766126e-05
    conditions_sizing.freestream.altitude                    = ones_1col* 10000.0
    conditions_sizing.freestream.gravity                     = ones_1col*9.81
    conditions_sizing.freestream.isentropic_expansion_factor = ones_1col*1.4
    conditions_sizing.freestream.Cp                          = 1.4*287.87/(1.4-1)
    conditions_sizing.freestream.R                           = 287.87
    conditions_sizing.freestream.speed_of_sound              = 299.53150968
    conditions_sizing.freestream.velocity                    = conditions_sizing.freestream.mach_number*conditions_sizing.freestream.speed_of_sound
    
    # propulsion conditions
    conditions_sizing.propulsion.throttle                    =  ones_1col*1.0

    # Setup Sizing
    state_sizing = Data()
    state_sizing.numerics = Data()
    state_sizing.conditions = conditions_sizing
    state_off_design=Data()
    state_off_design.numerics=Data()
    state_off_design.conditions=conditions


    # ------------------------------------------------------------------
    #   Ducted Fan Network
    # ------------------------------------------------------------------    

    #instantiate the ducted fan network
    ductedfan = SUAVE.Components.Energy.Networks.Ducted_Fan()
    ductedfan.tag = 'ductedfan'
    
    # setup
    ductedfan.bypass_ratio      = 0.0
    ductedfan.number_of_engines = 2.0
    ductedfan.engine_length     = 0.15
    ductedfan.nacelle_diameter  = 0.5
    
    # working fluid
    ductedfan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    # ------------------------------------------------------------------
    #   Component 1 - Ram
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    ductedfan.append(ram)

    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle   

    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 1.0
    inlet_nozzle.pressure_ratio        = 1.0
    
    # add to network
    ductedfan.append(inlet_nozzle)
    
    # ------------------------------------------------------------------
    #  Component 3 - Fan
    
    # instantiate 
    fan = SUAVE.Components.Energy.Converters.Fan()    
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 1.0
    fan.pressure_ratio        = 1.3    
    
    # add to network
    ductedfan.append(fan)

    # ------------------------------------------------------------------
    #  Component 4 - outlet_nozzle

    # instantiate
    fan_nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()    
    fan_nozzle.tag = 'fan_nozzle'
    
    # setup
    fan_nozzle.polytropic_efficiency = 1.0
    fan_nozzle.pressure_ratio        = 1.0    
    
    # add to network
    ductedfan.append(fan_nozzle)

    # ------------------------------------------------------------------
    #  Component 5 - Thrust
    # to compute thrust
    
    # instantiate
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='thrust'
    
    # setup
    thrust.total_design = 21191.50900196
    
    # add to network
    ductedfan.thrust    = thrust    

    #size the turbofan
    ducted_fan_sizing(ductedfan,0.5,10000.0)
    
    print("Design thrust ",ductedfan.design_thrust)
    print("Sealevel static thrust ",ductedfan.sealevel_static_thrust)
    
    results_design     = ductedfan(state_sizing)
    results_off_design = ductedfan(state_off_design)
    F                  = results_design.thrust_force_vector
    F_off_design       = results_off_design.thrust_force_vector
    
    # Test the model 
    # Specify the expected values
    expected = Data()
    expected.thrust = 21191.509001558181
    
    #error data function
    error =  Data()
    error.thrust_error = (F[0][0] -  expected.thrust)/expected.thrust
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)    
        
    return
    
if __name__ == '__main__':
    
    main()
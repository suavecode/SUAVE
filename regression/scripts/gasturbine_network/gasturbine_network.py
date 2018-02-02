# test_gasturbine_network.py
# 
# Created:  Anil Variyar, February 2015
# Modified: 
#        

""" create and evaluate a gas turbine network
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
#from SUAVE.Components.Energy.Gas_Turbine import Network
from SUAVE.Components.Energy.Networks.Turbofan import Turbofan
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing



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
    
    # --- Conditions        
    ones_1col = np.ones([1,1])       
    
    # setup conditions
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    '''
    conditions.frames       = Data()
    conditions.freestream   = Data()
    conditions.aerodynamics = Data()
    conditions.propulsion   = Data()
    conditions.weights      = Data()
    conditions.energies     = Data()
    '''
    #  self.conditions = conditions
   
    # freestream conditions
    conditions.freestream.mach_number                  = ones_1col*0.8
    conditions.freestream.pressure                     = ones_1col*20000.
    conditions.freestream.temperature                  = ones_1col*215.
    conditions.freestream.density                      = ones_1col*0.8

    conditions.freestream.dynamic_viscosity            = ones_1col* 0.000001475
    conditions.freestream.altitude                     = ones_1col* 10.
    conditions.freestream.gravity                      = ones_1col*9.81
    conditions.freestream.isentropic_expansion_factor  = ones_1col*1.4
    conditions.freestream.Cp                           = 1.4*287.87/(1.4-1)
    conditions.freestream.R                            = 287.87
    conditions.M                                       = conditions.freestream.mach_number 
    conditions.T                                       = conditions.freestream.temperature
    conditions.p                                       = conditions.freestream.pressure
    conditions.freestream.speed_of_sound               = ones_1col* np.sqrt(conditions.freestream.Cp/(conditions.freestream.Cp-conditions.freestream.R)*conditions.freestream.R*conditions.freestream.temperature) #300.
    conditions.freestream.velocity                     = conditions.M * conditions.freestream.speed_of_sound
    conditions.velocity                                = conditions.M * conditions.freestream.speed_of_sound
    conditions.q                                       = 0.5*conditions.freestream.density*conditions.velocity**2
    conditions.g0                                      = conditions.freestream.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle                     =  ones_1col*1.0
        
    # ------------------------------------------------------------------
    #   Design/sizing conditions
    # ------------------------------------------------------------------    
    
    # Conditions        
    ones_1col = np.ones([1,1])    
     
    # Setup conditions
    conditions_sizing = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    '''
    conditions_sizing.frames       = Data()
    conditions_sizing.freestream   = Data()
    conditions_sizing.aerodynamics = Data()
    conditions_sizing.propulsion   = Data()
    conditions_sizing.weights      = Data()
    conditions_sizing.energies     = Data()
    '''
    #  self.conditions = conditions

    # freestream conditions
    conditions_sizing.freestream.mach_number                 = ones_1col*0.8
    conditions_sizing.freestream.pressure                    = ones_1col*26499.73156529
    conditions_sizing.freestream.temperature                 = ones_1col*223.25186491
    conditions_sizing.freestream.density                     = ones_1col*0.41350854
    conditions_sizing.freestream.dynamic_viscosity           = ones_1col* 1.45766126e-05
    conditions_sizing.freestream.altitude                    = ones_1col* 10000.
    conditions_sizing.freestream.gravity                     = ones_1col*9.81
    conditions_sizing.freestream.isentropic_expansion_factor = ones_1col*1.4
    conditions_sizing.freestream.Cp                          = 1.4*287.87/(1.4-1)
    conditions_sizing.freestream.R                           = 287.87
    conditions_sizing.freestream.speed_of_sound              = 299.53150968
    conditions_sizing.freestream.velocity                    = conditions_sizing.freestream.mach_number * conditions_sizing.freestream.speed_of_sound
    
    # propulsion conditions
    conditions_sizing.propulsion.throttle                    =  ones_1col*1.0

    state_sizing                = Data()
    state_sizing.numerics       = Data()
    state_sizing.conditions     = conditions_sizing
    state_off_design            = Data()
    state_off_design.numerics   = Data()
    state_off_design.conditions = conditions


    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------    
    
    # Instantiate the gas turbine network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'
    
    # setup
    turbofan.bypass_ratio      = 5.4
    turbofan.number_of_engines = 2.0
    turbofan.engine_length     = 2.5
    turbofan.nacelle_diameter  = 1.580
    
    # working fluid
    turbofan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    
    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    turbofan.append(ram)


    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    
    # add to network
    turbofan.append(inlet_nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor
    
    # instantiate 
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'low_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 1.14    
    
    # add to network
    turbofan.append(compressor)

    
    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor
    
    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'high_pressure_compressor'
    
    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 13.415    
    
    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='low_pressure_turbine'
    
    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # add to network
    turbofan.append(turbine)
    
      
    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='high_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # add to network
    turbofan.append(turbine)
      
    
    # ------------------------------------------------------------------
    #  Component 7 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 0.99 
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1450
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    # add to network
    turbofan.append(combustor)

    
    # ------------------------------------------------------------------
    #  Component 8 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99    
    
    # add to network
    turbofan.append(nozzle)


    # ------------------------------------------------------------------
    #  Component 9 - Fan Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'fan_nozzle'

    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99    
    
    # add to network
    turbofan.append(nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 10 - Fan
    
    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    # add to network
    turbofan.append(fan)
    
    
    # ------------------------------------------------------------------
    #  Component 10 - Thrust
    
    # to compute thrust
    
    # instantiate
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='thrust'
    
    # setup
    thrust.total_design = 42383.01818423
    
    # add to network
    turbofan.thrust = thrust    
  
    numerics = Data()    
    eta=1.0
    
    #size the turbofan
    turbofan_sizing(turbofan,0.8,10000.0)
    
    print "Design thrust ",turbofan.design_thrust
    print "Sealevel static thrust ",turbofan.sealevel_static_thrust
    
    
    results_design     = turbofan(state_sizing)
    results_off_design = turbofan(state_off_design)
    F                  = results_design.thrust_force_vector
    mdot               = results_design.vehicle_mass_rate
    F_off_design       = results_off_design.thrust_force_vector
    mdot_off_design    = results_off_design.vehicle_mass_rate
    

    #Test the model 
    
    #Specify the expected values
    expected        = Data()
    expected.thrust = 42383.01818402065 
    expected.mdot   = 0.76425264
    
    #error data function
    error              =  Data()
    error.thrust_error = (F[0][0] -  expected.thrust)/expected.thrust
    error.mdot_error   = (mdot[0][0]-expected.mdot)/expected.mdot
    print error
    
    for k,v in error.items():
        assert(np.abs(v)<1e-6)    
    
    return
    
if __name__ == '__main__':
    
    main()
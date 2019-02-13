# gasturbine_network.py
# 
# Created:  Feb 2015, A. Variyar
# Modified: Sep 2018, W. Maier
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
    
    # Conditions        
    ones_1col = np.ones([1,1])       
    alt                                                = 10.0
    
    # Setup conditions
    planet     = SUAVE.Attributes.Planets.Earth()   
    atmosphere                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                        = atmosphere.compute_values(alt,0,True) 
    working_fluid                    = SUAVE.Attributes.Gases.Air()    
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
  
    # freestream conditions
    conditions.freestream.altitude                     = ones_1col*alt   
    conditions.freestream.mach_number                  = ones_1col*0.8
    conditions.freestream.pressure                     = ones_1col*atmo_data.pressure
    conditions.freestream.temperature                  = ones_1col*atmo_data.temperature
    conditions.freestream.density                      = ones_1col*atmo_data.density
    conditions.freestream.dynamic_viscosity            = ones_1col*atmo_data.dynamic_viscosity
    conditions.freestream.gravity                      = ones_1col*planet.compute_gravity(alt)
    conditions.freestream.isentropic_expansion_factor  = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)                                                                                             
    conditions.freestream.Cp                           = ones_1col*working_fluid.compute_cp(atmo_data.temperature,atmo_data.pressure)
    conditions.freestream.R                            = ones_1col*working_fluid.gas_specific_constant
    conditions.freestream.speed_of_sound               = ones_1col*atmo_data.speed_of_sound
    conditions.freestream.velocity                     = conditions.freestream.mach_number*conditions.freestream.speed_of_sound
    conditions.velocity                                = conditions.freestream.mach_number*conditions.freestream.speed_of_sound
    conditions.q                                       = 0.5*conditions.freestream.density*conditions.velocity**2
    conditions.g0                                      = conditions.freestream.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle                     =  ones_1col*1.0
        
    # ------------------------------------------------------------------
    #   Design/sizing conditions
    # ------------------------------------------------------------------    
    
    # Conditions        
    ones_1col = np.ones([1,1])    
    alt_size  = 10000.0
    # Setup conditions
    planet     = SUAVE.Attributes.Planets.Earth()   
    atmosphere                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                        = atmosphere.compute_values(alt_size,0,True) 
    working_fluid                    = SUAVE.Attributes.Gases.Air()    
    conditions_sizing = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

    # freestream conditions
    conditions_sizing.freestream.altitude                     = ones_1col*alt_size     
    conditions_sizing.freestream.mach_number                  = ones_1col*0.8
    conditions_sizing.freestream.pressure                     = ones_1col*atmo_data.pressure
    conditions_sizing.freestream.temperature                  = ones_1col*atmo_data.temperature
    conditions_sizing.freestream.density                      = ones_1col*atmo_data.density
    conditions_sizing.freestream.dynamic_viscosity            = ones_1col*atmo_data.dynamic_viscosity
    conditions_sizing.freestream.gravity                      = ones_1col*planet.compute_gravity(alt_size)
    conditions_sizing.freestream.isentropic_expansion_factor  = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)                                                                                             
    conditions_sizing.freestream.Cp                           = ones_1col*working_fluid.compute_cp(atmo_data.temperature,atmo_data.pressure)
    conditions_sizing.freestream.R                            = ones_1col*working_fluid.gas_specific_constant
    conditions_sizing.freestream.speed_of_sound               = ones_1col*atmo_data.speed_of_sound
    conditions_sizing.freestream.velocity                     = conditions_sizing.freestream.mach_number*conditions_sizing.freestream.speed_of_sound
    conditions_sizing.velocity                                = conditions_sizing.freestream.mach_number*conditions_sizing.freestream.speed_of_sound
    conditions_sizing.q                                       = 0.5*conditions_sizing.freestream.density*conditions_sizing.velocity**2
    conditions_sizing.g0                                      = conditions_sizing.freestream.gravity
    
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
    turbofan     = SUAVE.Components.Energy.Networks.Turbofan()
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
    
    print("Design thrust ",turbofan.design_thrust)
    print("Sealevel static thrust ",turbofan.sealevel_static_thrust)
    
    
    results_design     = turbofan(state_sizing)
    results_off_design = turbofan(state_off_design)
    F                  = results_design.thrust_force_vector
    mdot               = results_design.vehicle_mass_rate
    F_off_design       = results_off_design.thrust_force_vector
    mdot_off_design    = results_off_design.vehicle_mass_rate
    

    #Test the model 
    
    #Specify the expected values
    expected        = Data()
    expected.thrust = 42360.88505056
    expected.mdot   = 0.76399257
    
    #error data function
    error              =  Data()
    error.thrust_error = (F[0][0] -  expected.thrust)/expected.thrust
    error.mdot_error   = (mdot[0][0]-expected.mdot)/expected.mdot
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)    
    
    return
    
if __name__ == '__main__':
    
    main()
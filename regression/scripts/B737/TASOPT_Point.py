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

from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Input_Output.Results import  print_parasite_drag,  \
     print_compress_drag, \
     print_engine_data,   \
     print_mission_breakdown, \
     print_weight_breakdown
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # call the network function
    engine, state = energy_network()

    plot_engine_performance(engine, state)
    
    return

def plot_engine_performance(engine, state):
    results = Data()
    
    n = 25
    results.throttles = np.linspace(0.67, 1.0, n)
    results.sfcs = np.zeros(n)

    for idx, throttle in enumerate(results.throttles):
        state.conditions.propulsion.throttle = ones_1col = np.ones([1,1]) * throttle
        single_point_results= engine(state)
        # F                  = results_design.thrust_force_vector
        # mdot               = results_design.vehicle_mass_rate
        F       = single_point_results.thrust_force_vector
        mdot    = single_point_results.vehicle_mass_rate


        sfc = 3600. * mdot[0][0] / 0.1019715 / F[0][0]
        results.sfcs[idx] = sfc
        

        # print('mdot ' + str(mdot))
        # print('f_off_design ' + str(F[0][0]))
        # sfc = 3600. * mdot[0][0] / 0.1019715 / F[0][0]
        # print('SFC: ' + str(sfc))
        
    plt.plot(results.throttles, results.sfcs)
    plt.xlabel('Throttle setting')
    plt.ylabel('SFC (lb/lb_hr)')



def energy_network():
    
    # ------------------------------------------------------------------
    #   Evaluation Conditions
    # ------------------------------------------------------------------    
    
    # Conditions        
    ones_1col = np.ones([1,1])       
    alt                                                = 10.0 * Units.km
    
    # Setup conditions
    planet     = SUAVE.Attributes.Planets.Earth()   
    atmosphere                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                        = atmosphere.compute_values(alt,0,True) 
    working_fluid                    = SUAVE.Attributes.Gases.Air()    
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
  
    # freestream conditions
    conditions.freestream.altitude                     = ones_1col*alt   
    conditions.freestream.mach_number                  = ones_1col*0.78
    conditions.freestream.pressure                     = ones_1col*atmo_data.pressure
    conditions.freestream.temperature                  = ones_1col*atmo_data.temperature
    conditions.freestream.density                      = ones_1col*atmo_data.density
    conditions.freestream.dynamic_viscosity            = ones_1col*atmo_data.dynamic_viscosity
    conditions.freestream.gravity                      = ones_1col*planet.compute_gravity(alt)
    conditions.freestream.isentropic_expansion_factor  = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)                                                                                             
    conditions.freestream.Cp                           = ones_1col*working_fluid.compute_cp(atmo_data.temperature,atmo_data.pressure)
    conditions.freestream.specific_heat                = ones_1col*working_fluid.compute_cp(atmo_data.temperature,atmo_data.pressure)
    conditions.freestream.gamma                        = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)
    conditions.freestream.gas_specific_constant        = ones_1col*working_fluid.gas_specific_constant
    conditions.freestream.speed_of_sound               = ones_1col*atmo_data.speed_of_sound
    conditions.freestream.velocity                     = conditions.freestream.mach_number*conditions.freestream.speed_of_sound
    conditions.velocity                                = conditions.freestream.mach_number*conditions.freestream.speed_of_sound
    conditions.q                                       = 0.5*conditions.freestream.density*conditions.velocity**2
    conditions.g0                                      = conditions.freestream.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle                     =  ones_1col*0.73
        
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
    # conditions_sizing.freestream.altitude                     = ones_1col*alt_size     
    # conditions_sizing.freestream.mach_number                  = ones_1col*0.8
    # conditions_sizing.freestream.pressure                     = ones_1col*atmo_data.pressure
    # conditions_sizing.freestream.temperature                  = ones_1col*atmo_data.temperature
    # conditions_sizing.freestream.density                      = ones_1col*atmo_data.density
    # conditions_sizing.freestream.dynamic_viscosity            = ones_1col*atmo_data.dynamic_viscosity
    # conditions_sizing.freestream.gravity                      = ones_1col*planet.compute_gravity(alt_size)
    # conditions_sizing.freestream.isentropic_expansion_factor  = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)                                                                                             
    # conditions_sizing.freestream.Cp                           = ones_1col*working_fluid.compute_cp(atmo_data.temperature,atmo_data.pressure)
    # conditions_sizing.freestream.R                            = ones_1col*working_fluid.gas_specific_constant
    # conditions_sizing.freestream.speed_of_sound               = ones_1col*atmo_data.speed_of_sound
    # conditions_sizing.freestream.velocity                     = conditions_sizing.freestream.mach_number*conditions_sizing.freestream.speed_of_sound
    # conditions_sizing.freestream.gamma                        = ones_1col*working_fluid.compute_gamma(atmo_data.temperature,atmo_data.pressure)
    # conditions_sizing.velocity                                = conditions_sizing.freestream.mach_number*conditions_sizing.freestream.speed_of_sound
    # conditions_sizing.q                                       = 0.5*conditions_sizing.freestream.density*conditions_sizing.velocity**2
    # conditions_sizing.g0                                      = conditions_sizing.freestream.gravity
    
    # propulsion conditions
    # conditions_sizing.propulsion.throttle                    =  ones_1col*1.0

    # state_sizing                = Data()
    # state_sizing.numerics       = Data()
    # state_sizing.conditions     = conditions_sizing
    state_off_design            = Data()
    state_off_design.numerics   = Data()
    state_off_design.conditions = conditions

    # ------------------------------------------------------------------
    #   Turbofan Network
    # ------------------------------------------------------------------    

    #instantiate the gas turbine network
    gt_engine                   = SUAVE.Components.Energy.Networks.Turbofan_TASOPT_Net()
    #gt_engine                   = SUAVE.Components.Energy.Networks.Turbofan_TASOPT_Net_fsolve()
    gt_engine.tag               = 'turbofan'

    gt_engine.number_of_engines = 2.0
    gt_engine.bypass_ratio      = 5.4 #4.9 #5.4
    gt_engine.engine_length     = 2.71
    gt_engine.nacelle_diameter  = 2.05
    gt_engine.areas             = Data()
    gt_engine.areas.wetted = 1.    
    
    #compute engine areas
    Awet    = 1.1*np.pi*gt_engine.nacelle_diameter*gt_engine.engine_length 
    
    #Assign engine areas
    gt_engine.areas.wetted  = Awet
    
    
    
    #set the working fluid for the network
    working_fluid               = SUAVE.Attributes.Gases.Air

    #add working fluid to the network
    gt_engine.working_fluid = working_fluid


    # ------------------------------------------------------------------
    #Component 1 : ram,  to convert freestream static to stagnation quantities
    ram = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Ram()
    ram.tag = 'ram'


    #add ram to the network
    gt_engine.append(ram)


    # ------------------------------------------------------------------
    #Component 2 : inlet nozzle
    inlet_nozzle = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'

    inlet_nozzle.polytropic_efficiency = 1.0
    inlet_nozzle.pressure_ratio        = 0.999 #    turbofan.fan_nozzle_pressure_ratio     = 0.98     #0.98

    #add inlet nozzle to the network
    gt_engine.append(inlet_nozzle)


    # ------------------------------------------------------------------
    #Component 3 :low pressure compressor    
    low_pressure_compressor = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Compressor()    
    low_pressure_compressor.tag = 'low_pressure_compressor'

    low_pressure_compressor.polytropic_efficiency = 0.93 #0.93 #0.93
    low_pressure_compressor.pressure_ratio        = 1.94 #1.94 #2.04 #1.94

    low_pressure_compressor.design_polytropic_efficiency = 0.93
    low_pressure_compressor.design_pressure_ratio        = 1.94
    low_pressure_compressor.polytropic_efficiency  = 0.93
    low_pressure_compressor.pressure_ratio        = 1.94
    
    low_pressure_compressor.efficiency_map        = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Efficiency_Map()
    low_pressure_compressor.efficiency_map.design_polytropic_efficiency = 0.93
    low_pressure_compressor.efficiency_map.c                     = 3.0
    low_pressure_compressor.efficiency_map.C                     = 0.1
    low_pressure_compressor.efficiency_map.design_pressure_ratio = low_pressure_compressor.design_pressure_ratio
    
    low_pressure_compressor.speed_map             = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pressure_Ratio_Map()
    low_pressure_compressor.speed_map.pressure_ratio = low_pressure_compressor.pressure_ratio
    low_pressure_compressor.speed_map.a              = 3.0
    low_pressure_compressor.speed_map.b              = 0.85
    low_pressure_compressor.speed_map.k              = 0.03
    low_pressure_compressor.speed_map.design_pressure_ratio = low_pressure_compressor.design_pressure_ratio

    #add low pressure compressor to the network    
    gt_engine.append(low_pressure_compressor)


    # ------------------------------------------------------------------
    #Component 4 :high pressure compressor  
    high_pressure_compressor = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Compressor()    
    high_pressure_compressor.tag = 'high_pressure_compressor'

    high_pressure_compressor.polytropic_efficiency = 0.903 #0.93 #0.903
    high_pressure_compressor.pressure_ratio        = 9.36 #9.5 #9.36
    high_pressure_compressor.hub_to_tip_ratio      = 0.325 #9.36

    high_pressure_compressor.design_polytropic_efficiency = 0.903
    high_pressure_compressor.design_pressure_ratio        = 9.36
    high_pressure_compressor.polytropic_efficiency  = 0.903
    high_pressure_compressor.pressure_ratio        = 9.36
    
    high_pressure_compressor.efficiency_map        = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Efficiency_Map()
    high_pressure_compressor.efficiency_map.design_polytropic_efficiency = 0.903
    high_pressure_compressor.efficiency_map.c                     = 3.
    high_pressure_compressor.efficiency_map.C                     = 0.1
    high_pressure_compressor.efficiency_map.design_pressure_ratio = high_pressure_compressor.design_pressure_ratio
    
    high_pressure_compressor.speed_map             = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pressure_Ratio_Map()
    high_pressure_compressor.speed_map.pressure_ratio = high_pressure_compressor.pressure_ratio
    high_pressure_compressor.speed_map.a              = 1.5
    high_pressure_compressor.speed_map.b              = 5.0
    high_pressure_compressor.speed_map.k              = 0.03
    high_pressure_compressor.speed_map.design_pressure_ratio = high_pressure_compressor.design_pressure_ratio

    #add the high pressure compressor to the network    
    gt_engine.append(high_pressure_compressor)


    # ------------------------------------------------------------------
    #Component 5 :low pressure turbine  
    low_pressure_turbine = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Turbine()   
    low_pressure_turbine.tag='low_pressure_turbine'

    low_pressure_turbine.mechanical_efficiency = 0.99
    low_pressure_turbine.polytropic_efficiency = 0.882 #0.881

    #add low pressure turbine to the network    
    gt_engine.append(low_pressure_turbine)


    # ------------------------------------------------------------------
    #Component 5 :high pressure turbine  
    high_pressure_turbine = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Turbine()   
    high_pressure_turbine.tag='high_pressure_turbine'

    high_pressure_turbine.mechanical_efficiency = 0.99
    high_pressure_turbine.polytropic_efficiency = 0.874 #0.873

    #add the high pressure turbine to the network    
    gt_engine.append(high_pressure_turbine)


    # ------------------------------------------------------------------
    #Component 6 :combustor  
    cooling_flow = True
    if cooling_flow == True:
        combustor = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Cooling_Combustor()
        combustor.film_effectiveness_factor = 0.4
        combustor.weighted_stanton_number   = 0.035
        combustor.cooling_efficiency        = 0.7
        combustor.delta_temperature_streak  = 200.0
        combustor.metal_temperature         = 1400.0
        combustor.mixing_zone_start_mach_number = 0.8
        combustor.blade_row_exit_mach_number    = 0.8
        combustor.cooling_flow_velocity_ratio   = 0.9
    else:
        combustor = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Basic_Combustor()  
     
    combustor.tag = 'combustor'

    combustor.efficiency                = 0.983
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1585.0 #1585.0 #1480.0 #1485.0
    combustor.pressure_ratio            = 0.946
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    

    #add the combustor to the network    
    gt_engine.append(combustor)


    # ------------------------------------------------------------------
    #Component 7 :core nozzle
    core_nozzle = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Nozzle()   
    core_nozzle.tag = 'core_nozzle'

    core_nozzle.polytropic_efficiency = 1.0
    core_nozzle.pressure_ratio        = 0.99    

    #add the core nozzle to the network    
    gt_engine.append(core_nozzle)


    # ------------------------------------------------------------------
    #Component 8 :fan nozzle
    fan_nozzle = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Nozzle()   
    fan_nozzle.tag = 'fan_nozzle'

    fan_nozzle.polytropic_efficiency = 1.0
    fan_nozzle.pressure_ratio        = 0.99

    #add the fan nozzle to the network
    gt_engine.append(fan_nozzle)


    # ------------------------------------------------------------------
    #Component 9 : fan   
    fan = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Compressor()   
    fan.tag = 'fan'

    fan.design_polytropic_efficiency = 0.9 #0.9 #0.9
    fan.design_pressure_ratio        = 1.68
    fan.polytropic_efficiency  = 0.9
    fan.pressure_ratio        = 1.68 #1.68 #1.71 #1.68
    fan.hub_to_tip_ratio      = 0.325 #9.36
    
    fan.efficiency_map        = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Efficiency_Map()
    fan.efficiency_map.design_polytropic_efficiency = 0.9
    fan.efficiency_map.c                     = 3.0
    fan.efficiency_map.C                     = 0.1
    fan.efficiency_map.design_pressure_ratio = fan.design_pressure_ratio
    
    fan.speed_map             = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Pressure_Ratio_Map()
    fan.speed_map.pressure_ratio = fan.pressure_ratio
    fan.speed_map.a              = 3.0
    fan.speed_map.b              = 0.85
    fan.speed_map.k              = 0.03
    fan.speed_map.design_pressure_ratio = fan.design_pressure_ratio

    #add the fan to the network
    gt_engine.fan = fan  

    # Component 9.1
    core_exhaust = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Exhaust()
    core_exhaust.tag = 'core_exhaust'
    
    gt_engine.core_exhaust = core_exhaust
    
    # Component 9.2
    fan_exhaust = SUAVE.Components.Energy.Converters.Turbofan_TASOPT.Exhaust()
    fan_exhaust.tag = 'fan_exhaust'
    
    gt_engine.append(fan_exhaust) 

    # ------------------------------------------------------------------
    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust_TASOPT()       
    thrust.tag ='thrust'

    #total design thrust (includes all the engines)
    thrust.total_design             =  60000 * Units.N # 24366.8335 * Units.N #29450.0 * Units.N #26366.8335 * Units.N # 26520.3 * Units.N #  24000. * Units.N  #Newtons
    thrust.bypass_ratio = 5.4 #4.9 #5.4

    ##design sizing conditions
    altitude      = 35000.0*Units.ft
    mach_number   = 0.78 
    isa_deviation = 0.

    # add thrust to the network
    gt_engine.append(thrust)

    #size the turbofan
    gt_engine.unpack()
    gt_engine.size(mach_number,altitude)   


    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------

    throttles = np.linspace(.64, 1.0, 15)
    
    # results_design     = gt_engine(state_sizing)
    results_off_design = gt_engine(state_off_design)
    # F                  = results_design.thrust_force_vector
    # mdot               = results_design.vehicle_mass_rate
    F       = results_off_design.thrust_force_vector
    mdot    = results_off_design.vehicle_mass_rate
    

    print('mdot ' + str(mdot))
    print('f_off_design ' + str(F[0][0]))
    sfc = 3600. * mdot[0][0] / 0.1019715 / F[0][0]
    print('SFC: ' + str(sfc))
    #Test the model 
    
    #Specify the expected values
    expected        = Data()
    expected.thrust = 42360.88505056
    expected.mdot   = 0.76399257
    
    #error data function
    # error              =  Data()
    # error.thrust_error = (F[0][0] -  expected.thrust)/expected.thrust
    # error.mdot_error   = (mdot[0][0]-expected.mdot)/expected.mdot
    # print(error)
    
    # for k,v in list(error.items()):
    #     assert(np.abs(v)<1e-6)    
    
    return gt_engine, state_off_design
    
if __name__ == '__main__':
    
    main()
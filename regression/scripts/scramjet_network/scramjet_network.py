# scramjet_network.py
# 
# Created:  April 2018, W. Maier
# Modified: 
#        

""" create and evaluate a sramjet network
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data

import numpy as np

from SUAVE.Components.Energy.Networks.Scramjet import Scramjet
from SUAVE.Methods.Propulsion.scramjet_sizing import scramjet_sizing

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # call the network function
    energy_network()    

    return

# ----------------------------------------------------------------------
#   Energy Network
# ----------------------------------------------------------------------
def energy_network():
    
    # ------------------------------------------------------------------
    #   Evaluation Conditions
    # ------------------------------------------------------------------    
    
    # --- Conditions        
    ones_1col = np.ones([1,1])    
    
    # setup conditions
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    # freestream conditions
    eval                             = conditions.freestream
    eval.mach_number                 = ones_1col*4.5
    conditions.M                     = eval.mach_number
    eval.altitude                    = ones_1col*20000.
    
    atmosphere                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                        = atmosphere.compute_values(eval.altitude,0,True) 
    working_fluid                    = SUAVE.Attributes.Gases.Air()
                            
    eval.pressure                    = ones_1col*atmo_data.pressure
    eval.temperature                 = ones_1col*atmo_data.temperature
    eval.density                     = ones_1col*atmo_data.density
    eval.dynamic_viscosity           = ones_1col* atmo_data.dynamic_viscosity
    eval.gravity                     = ones_1col* SUAVE.Attributes.Planets.Earth().compute_gravity(eval.altitude)
    eval.isentropic_expansion_factor = working_fluid.compute_gamma(eval.temperature,eval.pressure)
    eval.Cp                          = working_fluid.compute_cp(eval.temperature,eval.pressure)                                                                               
    eval.R                           = working_fluid.gas_specific_constant
    eval.speed_of_sound              = ones_1col* atmo_data.speed_of_sound
    eval.velocity                    = conditions.M * eval.speed_of_sound
    conditions.velocity              = conditions.M * eval.speed_of_sound
    conditions.q                     = 0.5*eval.density*conditions.velocity**2
    conditions.g0                    = eval.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle   =  ones_1col*1.0

    # ------------------------------------------------------------------
    #   Design/sizing conditions
    # ------------------------------------------------------------------    
    
    # --- Conditions        
    ones_1col = np.ones([1,1])    
    
    # setup conditions
    conditions_sizing = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()

    # freestream conditions
    size                             = conditions_sizing.freestream
    size.mach_number                 = ones_1col*6.5
    conditions_sizing.M              = size.mach_number
    size.altitude                    = ones_1col*20000.  
    
    atmosphere                       = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                        = atmosphere.compute_values(size.altitude,0,True) 
    working_fluid                    = SUAVE.Attributes.Gases.Air()    

    size.pressure                    = ones_1col*atmo_data.pressure
    size.temperature                 = ones_1col*atmo_data.temperature
    size.density                     = ones_1col*atmo_data.density
    size.dynamic_viscosity           = ones_1col*atmo_data.dynamic_viscosity
    size.gravity                     = ones_1col*SUAVE.Attributes.Planets.Earth().compute_gravity(size.altitude)
    size.isentropic_expansion_factor = working_fluid.compute_gamma(size.temperature,size.pressure)
    size.Cp                          = working_fluid.compute_cp(size.temperature,size.pressure)                                                                               
    size.R                           = working_fluid.gas_specific_constant
    size.speed_of_sound              = ones_1col * atmo_data.speed_of_sound
    size.velocity                    = conditions_sizing.M * size.speed_of_sound
    conditions_sizing.velocity       = conditions_sizing.M * size.speed_of_sound
    conditions_sizing.q              = 0.5*size.density*conditions_sizing.velocity**2
    conditions_sizing.g0             = size.gravity
    
    # propulsion conditions
    conditions_sizing.propulsion.throttle = ones_1col*1.0

    state_sizing = Data()
    state_sizing.numerics = Data()
    state_sizing.conditions = conditions_sizing
    state_off_design=Data()
    state_off_design.numerics=Data()
    state_off_design.conditions=conditions

    # ------------------------------------------------------------------
    #  Ramjet Network
    # ------------------------------------------------------------------    
    
    # instantiate the ramjet network
    scramjet = SUAVE.Components.Energy.Networks.Scramjet()
    scramjet.tag = 'scramjet'
    
    # setup
    scramjet.number_of_engines = 1.0
    scramjet.engine_length     = 4.0
    scramjet.nacelle_diameter  = 0.3  * Units.meter
    scramjet.inlet_diameter    = 0.21 * Units.meter
    
    # working fluid
    scramjet.working_fluid = SUAVE.Attributes.Gases.Air()

    # ------------------------------------------------------------------
    #   Component 1 - Ram
    #   to convert freestream static to stagnation quantities
   
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    scramjet.append(ram)

    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency      = 0.90
    inlet_nozzle.pressure_ratio             = 1.0
    inlet_nozzle.compressibility_effects    = 3.0
    inlet_nozzle.compression_levels         = 3.0
    inlet_nozzle.theta                      = [0.10472,0.122173,0.226893]
    
    # add to network
    scramjet.append(inlet_nozzle)
      
    # ------------------------------------------------------------------
    #  Component 3 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 0.90 
    combustor.pressure_ratio            = 1.0
    combustor.area_ratio                = 2.0
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Liquid_H2()  
    combustor.burner_drag_coefficient   = 0.01
    combustor.fuel_equivalency_ratio    = 1.0
    
    # add to network
    scramjet.append(combustor)

    # ------------------------------------------------------------------
    #  Component 4 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Supersonic_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency    = 0.9
    nozzle.pressure_expansion_ratio = 1.1
    
    # add to network
    scramjet.append(nozzle)

    # ------------------------------------------------------------------
    #  Component 5 - Thrust
    
    # instantiate
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='thrust'
    
    # setup
    thrust.total_design = scramjet.number_of_engines*180000.0 * Units.N
    
    # add to network
    scramjet.thrust = thrust    

    #size the ramjet
    scramjet_sizing(scramjet,size.mach_number,size.altitude)
    
    print("Design thrust :",scramjet.design_thrust)
    print("Sealevel static thrust :",scramjet.sealevel_static_thrust)
    
    results_design     = scramjet(state_sizing)
    results_off_design = scramjet(state_off_design)
    F                  = results_design.thrust_force_vector
    mdot               = results_design.vehicle_mass_rate
    Isp                = results_design.specific_impulse   
    F_off_design       = results_off_design.thrust_force_vector
    mdot_off_design    = results_off_design.vehicle_mass_rate
    Isp_off_design     = results_off_design.specific_impulse
    
    #Specify the expected values
    expected        = Data()
    expected.thrust = 180000.0
    expected.mdot   = 7.8394948
    expected.Isp    = 2356.0590883
    
    #error data function
    error =  Data()
    
    error.thrust_error = (F[0][0] -  expected.thrust)/expected.thrust
    error.mdot_error   = (mdot[0][0] - expected.mdot)/expected.mdot
    error.Isp_error    = (Isp[0][0]- expected.Isp)/expected.Isp
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)    
    
    return
    
if __name__ == '__main__':
    
    main() 
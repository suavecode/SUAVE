# ramjet_network.py
# 
# Created:  Sep 2017, P. Goncalves
# Modified: Jan 2018, W. Maier
#        

""" create and evaluate a ramjet network
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data

import numpy as np

from SUAVE.Components.Energy.Networks.Ramjet import Ramjet
from SUAVE.Methods.Propulsion.ramjet_sizing import ramjet_sizing

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
    conditions.freestream.mach_number        = ones_1col*0.8
    conditions.freestream.pressure           = ones_1col*20000.
    conditions.freestream.temperature        = ones_1col*215.
    conditions.freestream.density            = ones_1col*0.8
    conditions.freestream.dynamic_viscosity  = ones_1col* 0.000001475
    conditions.freestream.altitude           = ones_1col* 10.
    conditions.freestream.gravity            = ones_1col*9.81
    conditions.freestream.gamma              = ones_1col*1.4
    conditions.freestream.Cp                 = 1.4*287.87/(1.4-1)
    conditions.freestream.R                  = 287.87
    conditions.M                             = conditions.freestream.mach_number 
    conditions.T                             = conditions.freestream.temperature
    conditions.p                             = conditions.freestream.pressure
    conditions.freestream.speed_of_sound     = ones_1col* np.sqrt(conditions.freestream.Cp/(conditions.freestream.Cp-conditions.freestream.R)*conditions.freestream.R*conditions.freestream.temperature) #300.
    conditions.freestream.velocity           = conditions.M * conditions.freestream.speed_of_sound
    conditions.velocity                      = conditions.M * conditions.freestream.speed_of_sound
    conditions.q                             = 0.5*conditions.freestream.density*conditions.velocity**2
    conditions.g0                            = conditions.freestream.gravity
    
    # propulsion conditions
    conditions.propulsion.throttle           =  ones_1col*1.0

    # ------------------------------------------------------------------
    #   Design/sizing conditions
    # ------------------------------------------------------------------    
    
    # --- Conditions        
    ones_1col = np.ones([1,1])    
    
    # setup conditions
    conditions_sizing = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()


    # freestream conditions
    conditions_sizing.freestream.mach_number        = ones_1col*2.5
    conditions_sizing.freestream.pressure           = ones_1col*26499.73156529
    conditions_sizing.freestream.temperature        = ones_1col*223.25186491
    conditions_sizing.freestream.density            = ones_1col*0.41350854
    conditions_sizing.freestream.dynamic_viscosity  = ones_1col* 1.45766126e-05 #*1.789*10**(-5)
    conditions_sizing.freestream.altitude           = ones_1col* 10000. #* 0.5
    conditions_sizing.freestream.gravity            = ones_1col*9.81
    conditions_sizing.freestream.gamma              = ones_1col*1.4
    conditions_sizing.freestream.Cp                 = 1.4*287.87/(1.4-1)
    conditions_sizing.freestream.R                  = 287.87
    conditions_sizing.freestream.speed_of_sound     = 299.96
    conditions_sizing.freestream.velocity           = conditions_sizing.freestream.mach_number * conditions_sizing.freestream.speed_of_sound
    
    # propulsion conditions
    conditions_sizing.propulsion.throttle           =  ones_1col*1.0

    state_sizing = Data()
    state_sizing.numerics = Data()
    state_sizing.conditions = conditions_sizing
    state_off_design=Data()
    state_off_design.numerics=Data()
    state_off_design.conditions=conditions

    # ------------------------------------------------------------------
    #  Ramjet Network
    # ------------------------------------------------------------------    
    
    # instantiate the gas turbine network
    ramjet = SUAVE.Components.Energy.Networks.Ramjet()
    ramjet.tag = 'ramjet'
    
    # setup
    ramjet.number_of_engines = 2.0
    ramjet.engine_length     = 12.0
    ramjet.nacelle_diameter  = 1.3 * Units.meter
    ramjet.inlet_diameter    = 1.1 * Units.meter
    
    # working fluid
    ramjet.working_fluid = SUAVE.Attributes.Gases.Air()

    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    ramjet.append(ram)

    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency      = 1.0
    inlet_nozzle.pressure_ratio             = 1.0
    inlet_nozzle.compressibility_effects    = True
    
    # add to network
    ramjet.append(inlet_nozzle)
      
    # ------------------------------------------------------------------
    #  Component 3 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 1.0 
    combustor.turbine_inlet_temperature = 2400.
    combustor.pressure_ratio            = 1.0
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()  
    combustor.rayleigh_analyses         = True
    
    # add to network
    ramjet.append(combustor)

    # ------------------------------------------------------------------
    #  Component 4 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Supersonic_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 1.0
    nozzle.pressure_ratio        = 1.0   
    
    # add to network
    ramjet.append(nozzle)

    # ------------------------------------------------------------------
    #  Component 5 - Thrust
    
    # instantiate
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='thrust'
    
    # setup
    thrust.total_design = ramjet.number_of_engines*169370.4652 * Units.N
    
    # add to network
    ramjet.thrust = thrust    

    #size the ramjet
    ramjet_sizing(ramjet,0.8,10000.0)
    
    print "Design thrust :",ramjet.design_thrust
    print "Sealevel static thrust :",ramjet.sealevel_static_thrust
    
    results_design     = ramjet(state_sizing)
    results_off_design = ramjet(state_off_design)
    F                  = results_design.thrust_force_vector
    mdot               = results_design.vehicle_mass_rate
    F_off_design       = results_off_design.thrust_force_vector
    mdot_off_design    = results_off_design.vehicle_mass_rate
    
    #Specify the expected values
    expected = Data()
    
    expected.thrust = 3025764.6957800589
    expected.mdot   = 201.72877231960354
    
    #error data function
    error =  Data()
    
    error.thrust_error = (F[0][0] -  expected.thrust)/expected.thrust
    error.mdot_error   = (mdot[0][0] - expected.mdot)/expected.mdot
    print error
    
    for k,v in error.items():
        assert(np.abs(v)<1e-6)    
    
    return
    
if __name__ == '__main__':
    
    main()
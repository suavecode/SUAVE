# Rocketdyne_F1.py
# 
# Created:  Feb 2018, W. Maier
# Modified: 
#        

""" Create and evaluate a F1 rocket engine
    Engine of main stage, S-1C, of Saturn V
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data

import numpy as np

from SUAVE.Components.Energy.Networks.Liquid_Rocket import Liquid_Rocket
from SUAVE.Methods.Propulsion.liquid_rocket_sizing  import liquid_rocket_sizing

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
    
    # setup conditions
    conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()   
    planet     = SUAVE.Attributes.Planets.Earth()
    ones_1col             = np.ones([1,1])   
    conditions.freestream = Data()
    conditions.propulsion = Data()
    
    
    # vacuum conditions
    vac                   = conditions.freestream
    vac.altitude          = ones_1col*0.0
    vac.gravity           = ones_1col*planet.sea_level_gravity
    vac.pressure          = ones_1col*0.0
        
    # propulsion conditions
    conditions.propulsion.throttle  =  ones_1col*1.0

    # ------------------------------------------------------------------
    #   Design/sizing conditions
    # ------------------------------------------------------------------    
    
    # setup conditions
    conditions_sls = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()      
    ones_1col                         = np.ones([1,1]) 
    conditions_sls.freestream         = Data()
    conditions_sls.propulsion         = Data()
    
    # freestream conditions
    SLS                               = conditions_sls.freestream
    SLS.altitude                      = ones_1col*0.0
    
    atmosphere                        = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data                         = atmosphere.compute_values(SLS.altitude,0,True) 
    
    SLS.pressure                      = ones_1col*atmo_data.pressure
    SLS.temperature                   = ones_1col*atmo_data.temperature
    SLS.density                       = ones_1col*atmo_data.density
    SLS.speed_of_sound                = ones_1col*atmo_data.speed_of_sound
    SLS.gravity                       = ones_1col*planet.sea_level_gravity
    
    # propulsion conditions
    conditions_sls.propulsion.throttle = ones_1col*1.0

    # setting states
    state_sls               = Data()
    state_sls.numerics      = Data()
    state_sls.conditions    = conditions_sls
    state_vacuum            = Data()
    state_vacuum.numerics   = Data()
    state_vacuum.conditions = conditions

    # ------------------------------------------------------------------
    #  F-1 Liquid Rocket Network
    # ------------------------------------------------------------------    
    
    # instantiate the ramjet network
    liquid_rocket = SUAVE.Components.Energy.Networks.Liquid_Rocket()
    liquid_rocket.tag = 'liquid_rocket'
    
    # setup
    liquid_rocket.number_of_engines = 1.0
    liquid_rocket.area_throat       = 0.6722
    liquid_rocket.contraction_ratio = 2.8956
    liquid_rocket.expansion_ratio   = 16.0
    
    # ------------------------------------------------------------------
    #   Component 1 - Combustor
     
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Rocket_Combustor()   
    combustor.tag = 'combustor'
    
    # setup  
    combustor.propellant_data                = SUAVE.Attributes.Propellants.LOX_RP1()
    combustor.inputs.combustion_pressure     = 7000000.0 
    
    # add to network
    liquid_rocket.append(combustor)

    # ------------------------------------------------------------------
    #  Component 2 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.de_Laval_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 1.0
    nozzle.expansion_ratio       = liquid_rocket.expansion_ratio
    nozzle.area_throat           = liquid_rocket.area_throat
    nozzle.pressure_ratio        = 1.0
    
    
    # add to network
    liquid_rocket.append(nozzle)

    # ------------------------------------------------------------------
    #  Component 4 - Thrust
    
    # instantiate
    thrust = SUAVE.Components.Energy.Processes.Rocket_Thrust()       
    thrust.tag ='thrust'
    
    # setup
    thrust.total_design  = liquid_rocket.number_of_engines*6770000 * Units.N
    thrust.ISP_design    = 263.0
    
    # add to network
    liquid_rocket.thrust = thrust 
    
    # size the rocket
    liquid_rocket_sizing(liquid_rocket,0.0)
    
    print("Design thrust :",liquid_rocket.thrust.total_design)
    print("Sealevel static thrust :",liquid_rocket.sealevel_static_thrust)
    
    results_SeaLevel   = liquid_rocket(state_sls)
    results_Vacuum     = liquid_rocket(state_vacuum)
    
    F_SeaLevel         = results_SeaLevel.thrust_force_vector
    mdot_SeaLevel      = results_SeaLevel.vehicle_mass_rate
    Isp_SeaLevel       = results_SeaLevel.specific_impulse
    
    F_Vacuum           = results_Vacuum.thrust_force_vector
    mdot_Vacuum        = results_Vacuum.vehicle_mass_rate
    Isp_Vacuum         = results_Vacuum.specific_impulse
    
    #Specify the expected values
    expected = Data()
    
    expected.thrust_SL = 7554319.11082433
    expected.mdot_SL   = 2607.71793795
    expected.Isp_SL    = 295.40241156
    
    expected.thrust_Vac = 8644089.75082433
    expected.mdot_Vac   = 2607.71793795
    expected.Isp_Vac    = 338.01655988  
    
    #error data function
    error =  Data()
    
    error.thrust_error_SL = (F_SeaLevel[0][0] -  expected.thrust_SL)/expected.thrust_SL
    error.mdot_error_SL   = (mdot_SeaLevel[0][0] - expected.mdot_SL)/expected.mdot_SL
    error.Isp_error_SL    = (Isp_SeaLevel[0][0]- expected.Isp_SL)/expected.Isp_SL
    
    error.thrust_error_Vac = (F_Vacuum[0][0] -  expected.thrust_Vac)/expected.thrust_Vac
    error.mdot_error_Vac   = (mdot_Vacuum[0][0] - expected.mdot_Vac)/expected.mdot_Vac
    error.Isp_error_Vac    = (Isp_Vacuum[0][0]- expected.Isp_Vac)/expected.Isp_Vac
    
    print(error)
    
    for k,v in error.items():
        assert(np.abs(v)<1e-6)    
    
    return
    
if __name__ == '__main__':
    
    main() 
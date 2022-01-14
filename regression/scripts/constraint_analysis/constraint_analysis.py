#constraint_analysis.py

# Created:  Dec 2020, S. Karpuk
# Modified: 


#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
import SUAVE.Analyses.Constraint_Analysis as Constraint_Analysis
from SUAVE.Plots.Performance.Plot_constraint_diagram import plot_constraint_diagram
import sys
import numpy as np
import matplotlib.pyplot as plt

from SUAVE.Core import Data, Units

sys.path.append('../Vehicles')



def main():

    # Sample jet airplane
    # ------------------------------------------------------------------

    # Define the vehicle and required constrain analysis parameters
    ca = Constraint_Analysis.Constraint_Analysis()

    plot_tag = False

    # Define default constraint analysis
    ca.analyses.takeoff.compute    = True
    ca.analyses.cruise.compute     = True
    ca.analyses.max_cruise.compute = True
    ca.analyses.landing.compute    = True
    ca.analyses.OEI_climb.compute  = True
    ca.analyses.turn.compute       = True
    ca.analyses.climb.compute      = True
    ca.analyses.ceiling.compute    = False

    # take-off
    ca.analyses.takeoff.runway_elevation = 0 * Units['meter']
    ca.analyses.takeoff.ground_run       = 1850 * Units['m']
    # climb
    ca.analyses.climb.altitude   = 35000 * Units['feet']
    ca.analyses.climb.airspeed   = 450   * Units['knots']
    ca.analyses.climb.climb_rate = 1.5   * Units['m/s']
    # OEI climb
    ca.analyses.OEI_climb.climb_speed_factor = 1.2
    #turn
    ca.analyses.turn.angle           = 15.0 * Units.degrees
    ca.analyses.turn.altitude        = 35000 * Units['feet']
    ca.analyses.turn.delta_ISA       = 0.0
    ca.analyses.turn.mach            = 0.78
    ca.analyses.turn.specific_energy = 0.0
    ca.analyses.turn.thrust_fraction = 1.0
    #cruise
    ca.analyses.cruise.altitude        = 35000 * Units['feet']  
    ca.analyses.cruise.mach            = 0.78
    ca.analyses.cruise.thrust_fraction = 0.8
    # max cruise
    ca.analyses.max_cruise.altitude        = 35000 * Units['feet']
    ca.analyses.max_cruise.delta_ISA       = 0.0
    ca.analyses.max_cruise.mach            = 0.82
    ca.analyses.max_cruise.thrust_fraction = 1.0
    # ceiling
    ca.analyses.ceiling.altitude  = 39000 * Units['feet']
    ca.analyses.ceiling.delta_ISA = 0.0
    ca.analyses.ceiling.mach      = 0.78
    # landing
    ca.analyses.landing.ground_roll = 1400 * Units['m']  


    # Default aircraft properties
    vehicle = SUAVE.Vehicle()

    # geometry
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio             = 10.18
    wing.thickness_to_chord       = 0.1
    wing.taper                    = 0.1
    wing.sweeps.quarter_chord     = 35 * Units.degrees

    flap                          = SUAVE.Components.Wings.Control_Surfaces.Flap()
    flap.tag                      = 'flap'
    flap.configuration_type       = 'double_slotted'
    wing.append_control_surface(flap)

    # add to vehicle
    vehicle.append_component(wing)

    # add energy network
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.number_of_engines = 2.0
    turbofan.bypass_ratio      = 5.4


    # add  gas turbine network turbofan to the vehicle
    vehicle.append_component(turbofan)


    ca.engine.method         = "Mattingly"            # defines turbofan constraint analysis method 
    ca.engine.throttle_ratio = 1.0

    # Define aerodynamics for the constraint analysis(an example with pre-defined max lift coefficients)
    ca.aerodynamics.cd_takeoff     = 0.044
    ca.aerodynamics.cl_takeoff     = 0.6
    ca.aerodynamics.cd_min_clean   = 0.017
    ca.aerodynamics.cl_max_takeoff = 2.1
    ca.aerodynamics.cl_max_landing = 2.4

    ca.design_point_type = 'maximum wing loading'
    # ---------------------------------------------------------------------------------------------------------------

    # run the constraint diagrams for various engine models

    constraint_results = ca.create_constraint_diagram(vehicle)
    plot_constraint_diagram(constraint_results,vehicle,plot_tag)

    jet_WS_Matt = constraint_results.des_wing_loading
    jet_TW_Matt = constraint_results.des_thrust_to_weight

    ca.engine.method   = "Scholz"
    constraint_results = ca.create_constraint_diagram(vehicle)

    jet_WS_Scholz = constraint_results.des_wing_loading
    jet_TW_Scholz = constraint_results.des_thrust_to_weight

    ca.engine.method   = "Howe"
    constraint_results = ca.create_constraint_diagram(vehicle)

    jet_WS_Howe = constraint_results.des_wing_loading
    jet_TW_Howe = constraint_results.des_thrust_to_weight
    
    ca.engine.method   = "Bartel"
    constraint_results = ca.create_constraint_diagram(vehicle)

    jet_WS_Bartel = constraint_results.des_wing_loading
    jet_TW_Bartel = constraint_results.des_thrust_to_weight

 

    # Sample propeller airplanes
    # ------------------------------------------------------------------
    # Define the vehicle and required constrain analysis parameters

    ca = Constraint_Analysis.Constraint_Analysis()

    # Define default constraint analysis
    ca.analyses.takeoff.compute     = True
    ca.analyses.cruise.compute      = True
    ca.analyses.max_cruise.compute  = True 
    ca.analyses.landing.compute     = True
    ca.analyses.OEI_climb.compute   = True
    ca.analyses.turn.compute        = True
    ca.analyses.climb.compute       = True
    ca.analyses.ceiling.compute     = True

    # take-off
    ca.analyses.takeoff.runway_elevation    = 0 * Units['meter']
    ca.analyses.takeoff.ground_run          = 450 * Units['m']
    # climb
    ca.analyses.climb.altitude              = 0.0   * Units['meter']
    ca.analyses.climb.climb_rate            = 3.66   * Units['m/s']
    # cruise
    ca.analyses.cruise.altitude             = 3000 * Units['meter']  
    ca.analyses.cruise.mach                 = 0.15
    ca.analyses.cruise.thrust_fraction      = 1
    # max cruise
    ca.analyses.max_cruise.altitude        = 4000 * Units['meter'] 
    ca.analyses.max_cruise.delta_ISA       = 0.0
    ca.analyses.max_cruise.mach            = 0.25
    ca.analyses.max_cruise.thrust_fraction = 1.0
    # turn
    ca.analyses.turn.angle                  = 15    * Units['degrees']
    ca.analyses.turn.altitude               = 3000  * Units['meter']   
    ca.analyses.turn.mach                   = 0.2
    ca.analyses.turn.thrust_fraction        = 1
    # OEI climb
    ca.analyses.OEI_climb.climb_speed_factor = 1.2
    # ceiling
    ca.analyses.ceiling.altitude            = 4000 * Units['meter'] 
    ca.analyses.ceiling.mach                = 0.15
    # landing
    ca.analyses.landing.ground_roll         = 120 * Units['m']   

    # propeller
    ca.propeller.takeoff_efficiency   = 0.5
    ca.propeller.climb_efficiency     = 0.8
    ca.propeller.cruise_efficiency    = 0.85
    ca.propeller.turn_efficiency      = 0.85
    ca.propeller.ceiling_efficiency   = 0.85
    ca.propeller.OEI_climb_efficiency = 0.5

    # Define aerodynamics (an example case with max lift calculation for differnet flap settions)
    ca.aerodynamics.cd_takeoff     = 0.04
    ca.aerodynamics.cl_takeoff     = 0.6
    ca.aerodynamics.cd_min_clean   = 0.02

    ca.design_point_type = 'minimum power-to-weight'


    vehicle  = SUAVE.Vehicle()
    wing     = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio                 = 7.32
    wing.thickness_to_chord           = 0.13
    wing.taper                        = 1.0
    wing.sweeps.quarter_chord         = 0 * Units.degrees

    # add to vehicle
    vehicle.append_component(wing)


    # set configs up
    configs = configs_setup(vehicle)

    # run the constraint diagram for various engine and flap types
    config             = configs.turboprop_double_slotted_fowler
    constraint_results = ca.create_constraint_diagram(config)
    turboprop_WS       = constraint_results.des_wing_loading
    turboprop_TW       = constraint_results.des_thrust_to_weight
    plot_constraint_diagram(constraint_results,config,plot_tag,filename ='constraint_diagram_turboprop')

    config             = configs.turboprop_double_slotted
    constraint_results = ca.create_constraint_diagram(config)
    turboprop1_WS      = constraint_results.des_wing_loading
    turboprop1_TW      = constraint_results.des_thrust_to_weight
    plot_constraint_diagram(constraint_results,config,plot_tag,filename ='constraint_diagram_turboprop1')

    config             = configs.battery_prop_single_slotted_fowler
    constraint_results = ca.create_constraint_diagram(config)
    electric_WS        = constraint_results.des_wing_loading
    electric_TW        = constraint_results.des_thrust_to_weight
    plot_constraint_diagram(constraint_results,config,plot_tag,filename ='constraint_diagram_electric')


    config             = configs.ice_single_slotted
    constraint_results = ca.create_constraint_diagram(config)
    piston_WS          = constraint_results.des_wing_loading
    piston_TW          = constraint_results.des_thrust_to_weight
    plot_constraint_diagram(constraint_results,config,plot_tag,filename ='constraint_diagram_piston')


    # expected values
    turboprop_WS_true       = 825.9344674
    turboprop_TW_true       = 10.3861192
    turboprop1_WS_true      = 780.222339
    turboprop1_TW_true      = 10.753027
    piston_WS_true          = 629.027516
    piston_TW_true          = 15.0517943
    electric_TW_true        = 9.221734
    electric_WS_true        = 658.35356
    jet_WS_Matt_true        = 6412.752
    jet_TW_Matt_true        = 0.3006469
    jet_WS_Scholz_true      = 6412.752
    jet_TW_Scholz_true      = 0.21992447
    jet_WS_Howe_true        = 6412.752
    jet_TW_Howe_true        = 0.27025933
    jet_WS_Bartel_true      = 6412.752
    jet_TW_Bartel_true      = 0.277663710


    err_turboprop_WS        = (turboprop_WS - turboprop_WS_true)/turboprop_WS_true
    err_turboprop_TW        = (turboprop_TW - turboprop_TW_true)/turboprop_TW_true
    err_turboprop1_WS       = (turboprop1_WS - turboprop1_WS_true)/turboprop1_WS_true
    err_turboprop1_TW       = (turboprop1_TW - turboprop1_TW_true)/turboprop1_TW_true
    err_piston_WS           = (piston_WS - piston_WS_true)/piston_WS_true
    err_piston_TW           = (piston_TW - piston_TW_true)/piston_TW_true
    err_electric_WS         = (electric_WS - electric_WS_true)/electric_WS_true
    err_electric_TW         = (electric_TW - electric_TW_true)/electric_TW_true
    err_jet_WS_Matt         = (jet_WS_Matt  - jet_WS_Matt_true)/jet_WS_Matt_true
    err_jet_TW_Matt         = (jet_TW_Matt  - jet_TW_Matt_true)/jet_TW_Matt_true
    err_jet_WS_Scholz       = (jet_WS_Scholz  - jet_WS_Scholz_true)/jet_WS_Scholz_true
    err_jet_TW_Scholz       = (jet_TW_Scholz  - jet_TW_Scholz_true)/jet_TW_Scholz_true
    err_jet_WS_Howe         = (jet_WS_Howe  - jet_WS_Howe_true)/jet_WS_Howe_true
    err_jet_TW_Howe         = (jet_TW_Howe - jet_TW_Howe_true)/jet_TW_Howe_true
    err_jet_WS_Bartel       = (jet_WS_Bartel  - jet_WS_Bartel_true)/jet_WS_Bartel_true
    err_jet_TW_Bartel       = (jet_TW_Bartel - jet_TW_Bartel_true)/jet_TW_Bartel_true
 

    print('Calculated values:')
    print('Turboprop           : W/S = ' + str(turboprop_WS) + ', P/W = ' + str(turboprop_TW))
    print('Turboprop1          : W/S = ' + str(turboprop1_WS)  + ', P/W = ' + str(turboprop1_TW))
    print('Piston              : W/S = ' + str(piston_WS) + ', P/W = ' + str(piston_TW))
    print('Electric            : W/S = ' + str(electric_WS) + ', P/W = ' + str(electric_TW))
    print('Turbofan, Mattingly : W/S = ' + str(jet_WS_Matt) + ', T/W = ' + str(jet_TW_Matt))    
    print('Turbofan, Scholz    : W/S = ' + str(jet_WS_Scholz) + ', T/W = ' + str(jet_TW_Scholz))   
    print('Turbofan, Howe      : W/S = ' + str(jet_WS_Howe) + ', T/W = ' + str(jet_TW_Howe))  
    print('Turbofan, Bartel    : W/S = ' + str(jet_WS_Bartel) + ', T/W = ' + str(jet_TW_Bartel)) 
    
    print('Expected values:')
    print('Turboprop           : W/S = ' + str(turboprop_WS_true)  + ', P/W = ' + str(turboprop_TW_true))
    print('Turboprop1          : W/S = ' + str(turboprop1_WS_true)  + ', P/W = ' + str(turboprop1_TW_true))
    print('Piston              : W/S = ' + str(piston_WS_true)     + ', P/W = ' + str(piston_TW_true))
    print('Electric air        : W/S = ' + str(electric_WS_true)   + ', P/W = ' + str(electric_TW_true))
    print('Turbofan, Mattingly : W/S = ' + str(jet_WS_Matt_true)   + ', T/W = ' + str(jet_TW_Matt_true))    
    print('Turbofan, Scholz    : W/S = ' + str(jet_WS_Scholz_true) + ', T/W = ' + str(jet_TW_Scholz_true))   
    print('Turbofan, Howe      : W/S = ' + str(jet_WS_Howe_true)   + ', T/W = ' + str(jet_TW_Howe_true))  
    print('Turbofan, Bartel    : W/S = ' + str(jet_WS_Bartel_true) + ', T/W = ' + str(jet_TW_Bartel_true)) 


    err = Data()
    err.turboprop_WS_error      = err_turboprop_WS
    err.turboprop1_WS_error     = err_turboprop1_WS
    err.turboprop_TW_error      = err_turboprop_TW
    err.turboprop1_TW_error     = err_turboprop1_TW
    err.piston_WS_error         = err_piston_WS
    err.piston_TW_error         = err_piston_TW
    err.electric_air_WS_error   = err_electric_WS
    err.electric_air_TW_error   = err_electric_TW

    err.jet_WS_Matt_error       = err_jet_WS_Matt
    err.jet_TW_Matt_error       = err_jet_TW_Matt
    err.jet_WS_Scholz_error     = err_jet_WS_Scholz
    err.jet_TW_Scholz_error     = err_jet_TW_Scholz
    err.jet_WS_Howe_error       = err_jet_WS_Howe
    err.jet_TW_Howe_error       = err_jet_TW_Howe
    err.jet_WS_Bartel_error     = err_jet_WS_Bartel
    err.jet_TW_Bartel_error     = err_jet_TW_Bartel

    
    print('Errors:')
    print('Turboprop           : W/S = ' + str(err_turboprop_WS) + ', P/W = ' + str(err_turboprop_TW))
    print('Turboprop1          : W/S = ' + str(err_turboprop1_WS) + ', P/W = ' + str(err_turboprop1_TW))
    print('Piston              : W/S = ' + str(err_piston_WS) + ', P/W = ' + str(err_piston_TW))
    print('Electric            : W/S = ' + str(err_electric_WS) + ', P/W = ' + str(err_electric_TW))
    print('Turbofan, Mattingly : W/S = ' + str(err_jet_WS_Matt) + ', T/W = ' + str(err_jet_TW_Matt))    
    print('Turbofan, Scholz    : W/S = ' + str(err_jet_WS_Scholz) + ', T/W = ' + str(err_jet_TW_Scholz))   
    print('Turbofan, Howe      : W/S = ' + str(err_jet_WS_Howe) + ', T/W = ' + str(err_jet_TW_Howe))  
    print('Turbofan, Bartel    : W/S = ' + str(err_jet_WS_Bartel) + ', T/W = ' + str(err_jet_TW_Bartel)) 



    for k,v in list(err.items()):
        assert(np.abs(v)<1E-6)    


def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Base Configuration
    # ------------------------------------------------------------------
    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    # ------------------------------------------------------------------
    #   ICE with Single-slotted flaps 
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'ICE_single_slotted' 

    flap                          = SUAVE.Components.Wings.Control_Surfaces.Flap()
    flap.tag                      = 'flap'
    flap.configuration_type       = 'single_slotted'

    config.wings['main_wing'].append_control_surface(flap)

    # engine
    internal_comb_eng = SUAVE.Components.Energy.Networks.Internal_Combustion_Propeller()
    internal_comb_eng.number_of_engines = 2.0
    config.networks.append(internal_comb_eng)

    configs.append(config)

    # ------------------------------------------------------------------
    #   Battery propeller with Single-slotted Fowler flaps
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'Battery_prop_single_slotted_Fowler' 

    flap                          = SUAVE.Components.Wings.Control_Surfaces.Flap()
    flap.tag                      = 'flap'
    flap.configuration_type       = 'single_slotted_Fowler'

    config.wings['main_wing'].append_control_surface(flap)

    # engine
    battery_prop = SUAVE.Components.Energy.Networks.Battery_Propeller()
    battery_prop.number_of_engines = 2.0
    config.networks.append(battery_prop)
    
    configs.append(config)

    # ------------------------------------------------------------------
    #   Turboprop with Double-slotted flaps
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'Turboprop_double_slotted' 

    flap                          = SUAVE.Components.Wings.Control_Surfaces.Flap()
    flap.tag                      = 'flap'
    flap.configuration_type       = 'double_slotted'

    config.wings['main_wing'].append_control_surface(flap)

    # engine
    turboprop = SUAVE.Components.Energy.Networks.Turboprop()
    turboprop.number_of_engines = 2.0
    config.networks.append(turboprop)
    
    configs.append(config)

    # ------------------------------------------------------------------
    #   Tirboprop with Double-slotted Fowler
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'Turboprop_double_slotted_Fowler' 

    flap                          = SUAVE.Components.Wings.Control_Surfaces.Flap()
    flap.tag                      = 'flap'
    flap.configuration_type       = 'double_slotted_Fowler'

    config.wings['main_wing'].append_control_surface(flap)

    # engine
    turboprop = SUAVE.Components.Energy.Networks.Turboprop()
    turboprop.number_of_engines = 2.0
    config.networks.append(turboprop)
    
    configs.append(config)


    return configs

    
if __name__ == '__main__':
    main()




#constraint_analysis.py

# Created:  Dec 2020, S. Karpuk
# Modified: 


#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
import SUAVE.Analyses.Constraint_Analysis as Constraint_Analysis
from SUAVE.Input_Output.Results.plot_constraint_diagram import plot_constraint_diagram
import sys
import numpy as np
import matplotlib.pyplot as plt

from SUAVE.Core import Data, Units

sys.path.append('../Vehicles')



def main():

    # Sample jet airplane
    # ------------------------------------------------------------------

    # Define the vehicle and required constrain analysis parameters
    vehicle = SUAVE.Vehicle()

    # Define default constraint analysis
    vehicle.constraints.analyses.takeoff.compute    = True
    vehicle.constraints.analyses.cruise.compute     = True
    vehicle.constraints.analyses.max_cruise.compute = True
    vehicle.constraints.analyses.landing.compute    = True
    vehicle.constraints.analyses.OEI_climb.compute  = True
    vehicle.constraints.analyses.turn.compute       = True
    vehicle.constraints.analyses.climb.compute      = True
    vehicle.constraints.analyses.ceiling.compute    = False

    # take-off
    vehicle.constraints.analyses.takeoff.runway_elevation = 0 * Units['meter']
    vehicle.constraints.analyses.takeoff.ground_run       = 1550 * Units['m']
    # climb
    vehicle.constraints.analyses.climb.altitude   = 35000 * Units['feet']
    vehicle.constraints.analyses.climb.airspeed   = 450   * Units['knots']
    vehicle.constraints.analyses.climb.climb_rate = 2.0   * Units['m/s']
    # OEI climb
    vehicle.constraints.analyses.OEI_climb.climb_speed_factor = 1.2
    #turn
    vehicle.constraints.analyses.turn.angle           = 15.0 * Units.degrees
    vehicle.constraints.analyses.turn.altitude        = 35000 * Units['feet']
    vehicle.constraints.analyses.turn.delta_ISA       = 0.0
    vehicle.constraints.analyses.turn.mach            = 0.78
    vehicle.constraints.analyses.turn.specific_energy = 0.0
    vehicle.constraints.analyses.turn.thrust_fraction = 1.0
    #cruise
    vehicle.constraints.analyses.cruise.altitude        = 35000 * Units['feet']  
    vehicle.constraints.analyses.cruise.mach            = 0.78
    vehicle.constraints.analyses.cruise.thrust_fraction = 0.8
    # max cruise
    vehicle.constraints.analyses.max_cruise.altitude        = 35000 * Units['feet']
    vehicle.constraints.analyses.max_cruise.delta_ISA       = 0.0
    vehicle.constraints.analyses.max_cruise.mach            = 0.82
    vehicle.constraints.analyses.max_cruise.thrust_fraction = 1.0
    # ceiling
    vehicle.constraints.analyses.ceiling.altitude  = 39000 * Units['feet']
    vehicle.constraints.analyses.ceiling.delta_ISA = 0.0
    vehicle.constraints.analyses.ceiling.mach      = 0.78
    # landing
    vehicle.constraints.analyses.landing.ground_roll = 1400 * Units['m']  


    # Default aircraft properties
    # geometry
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio                 = 10.18
    wing.thickness_to_chord           = 0.1
    wing.taper                        = 0.1
    wing.sweeps.quarter_chord         = 35 * Units.degrees

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

    vehicle.constraints.engine.type           = 'turbofan'             # defines the engine type specifically for the constraint analysis
    vehicle.constraints.engine.method         = "Mattingly"            # defines turbofan constraint analysis method 
    vehicle.constraints.engine.throttle_ratio = 1.0


    # Define aerodynamics for the constraint analysis(an example with pre-defined max lift coefficients)
    vehicle.constraints.aerodynamics.cd_takeoff     = 0.044
    vehicle.constraints.aerodynamics.cl_takeoff     = 0.6
    vehicle.constraints.aerodynamics.cd_min_clean   = 0.017
    vehicle.constraints.aerodynamics.cl_max_takeoff = 2.1
    vehicle.constraints.aerodynamics.cl_max_landing = 2.4

    vehicle.constraints.design_point_type = 'maximum wing loading'
    # ---------------------------------------------------------------------------------------------------------------


    # Define analysis
    ca = Constraint_Analysis.Constraint_Analysis()
    # run the constraint diagrams for various engine models

    constraint_results = ca.create_constraint_diagram(vehicle)
    plot_constraint_diagram(constraint_results,vehicle.constraints.plot_tag,vehicle.constraints.engine.type)

    jet_WS_Matt = constraint_results.des_wing_loading
    jet_TW_Matt = constraint_results.des_thrust_to_weight

    vehicle.constraints.engine.method   = "Scholz"
    constraint_results = ca.create_constraint_diagram(vehicle)

    jet_WS_Scholz = constraint_results.des_wing_loading
    jet_TW_Scholz = constraint_results.des_thrust_to_weight

    vehicle.constraints.engine.method   = "Howe"
    constraint_results = ca.create_constraint_diagram(vehicle)

    jet_WS_Howe = constraint_results.des_wing_loading
    jet_TW_Howe = constraint_results.des_thrust_to_weight
    
    vehicle.constraints.engine.method   = "Bartel"
    constraint_results = ca.create_constraint_diagram(vehicle)

    jet_WS_Bartel = constraint_results.des_wing_loading
    jet_TW_Bartel = constraint_results.des_thrust_to_weight

 

    # Sample propeller airplanes
    # ------------------------------------------------------------------
    ca = Constraint_Analysis.Constraint_Analysis()

    # Define default constraint analysis
    vehicle.constraints.analyses.takeoff.compute     = True
    vehicle.constraints.analyses.cruise.compute      = True
    vehicle.constraints.analyses.max_cruise.compute  = True 
    vehicle.constraints.analyses.landing.compute     = True
    vehicle.constraints.analyses.OEI_climb.compute   = True
    vehicle.constraints.analyses.turn.compute        = True
    vehicle.constraints.analyses.climb.compute       = True
    vehicle.constraints.analyses.ceiling.compute     = True

    # take-off
    vehicle.constraints.analyses.takeoff.runway_elevation    = 0 * Units['meter']
    vehicle.constraints.analyses.takeoff.ground_run          = 450 * Units['m']
    # climb
    vehicle.constraints.analyses.climb.altitude              = 0.0   * Units['meter']
    vehicle.constraints.analyses.climb.climb_rate            = 3.66   * Units['m/s']
    # cruise
    vehicle.constraints.analyses.cruise.altitude             = 3000 * Units['meter']  
    vehicle.constraints.analyses.cruise.mach                 = 0.15
    vehicle.constraints.analyses.cruise.thrust_fraction      = 1
    # max cruise
    vehicle.constraints.analyses.max_cruise.altitude        = 4000 * Units['meter'] 
    vehicle.constraints.analyses.max_cruise.delta_ISA       = 0.0
    vehicle.constraints.analyses.max_cruise.mach            = 0.25
    vehicle.constraints.analyses.max_cruise.thrust_fraction = 1.0
    # turn
    vehicle.constraints.analyses.turn.angle                  = 15    * Units['degrees']
    vehicle.constraints.analyses.turn.altitude               = 3000  * Units['meter']   
    vehicle.constraints.analyses.turn.mach                   = 0.2
    vehicle.constraints.analyses.turn.thrust_fraction        = 1
    # OEI climb
    vehicle.constraints.analyses.OEI_climb.climb_speed_factor = 1.2
    # ceiling
    vehicle.constraints.analyses.ceiling.altitude            = 4000 * Units['meter'] 
    vehicle.constraints.analyses.ceiling.mach                = 0.15
    # landing
    vehicle.constraints.analyses.landing.ground_roll         = 120 * Units['m']   

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.aspect_ratio                 = 7.32
    wing.thickness_to_chord           = 0.13
    wing.taper                        = 1.0
    wing.sweeps.quarter_chord         = 0 * Units.degrees

    flap                          = SUAVE.Components.Wings.Control_Surfaces.Flap()
    flap.tag                      = 'flap'
    flap.configuration_type       = 'single_slotted'
    wing.append_control_surface(flap)

    # add to vehicle
    vehicle.append_component(wing)

    # engine
    network = SUAVE.Components.Energy.Networks.Internal_Combustion_Propeller()
    network.number_of_engines = 2.0

    # propeller
    vehicle.constraints.propeller.takeoff_efficiency   = 0.5
    vehicle.constraints.propeller.climb_efficiency     = 0.8
    vehicle.constraints.propeller.cruise_efficiency    = 0.85
    vehicle.constraints.propeller.turn_efficiency      = 0.85
    vehicle.constraints.propeller.ceiling_efficiency   = 0.85
    vehicle.constraints.propeller.OEI_climb_efficiency = 0.5

    # Define aerodynamics (an example case with max lift calculation for differnet flap settions)
    vehicle.constraints.aerodynamics.cd_takeoff     = 0.04
    vehicle.constraints.aerodynamics.cl_takeoff     = 0.6
    vehicle.constraints.aerodynamics.cd_min_clean   = 0.02

    vehicle.constraints.design_point_type = 'minimum power-to-weight'

    # run the constraint diagram for various engine ad flap types
    vehicle.constraints.engine.type = 'turboprop'
    constraint_results = ca.create_constraint_diagram(vehicle)

    turboprop_WS = constraint_results.des_wing_loading
    turboprop_TW = constraint_results.des_thrust_to_weight

    vehicle.constraints.engine.type = 'piston'
    vehicle.control_surfaces.flap.configuration_type  = 'double_slotted'
    constraint_results = ca.create_constraint_diagram(vehicle)
    plot_constraint_diagram(constraint_results,vehicle.constraints.plot_tag,vehicle.constraints.engine.type,filename ='constraint_diagram_piston')

    piston_WS = constraint_results.des_wing_loading
    piston_TW = constraint_results.des_thrust_to_weight

    vehicle.constraints.engine.type = 'electric air-cooled'
    constraint_results = ca.create_constraint_diagram(vehicle)
    vehicle.control_surfaces.flap.configuration_type  = 'single_slotted_Fowler'
    plot_constraint_diagram(constraint_results,vehicle.constraints.plot_tag,vehicle.constraints.engine.type,filename ='constraint_diagram_electric_air')

    electric_air_WS = constraint_results.des_wing_loading
    electric_air_TW = constraint_results.des_thrust_to_weight

    vehicle.constraints.engine.type = 'electric liquid-cooled'
    vehicle.control_surfaces.flap.configuration_type  = 'double_slotted_Fowler'
    constraint_results = ca.create_constraint_diagram(vehicle)
    plot_constraint_diagram(constraint_results,vehicle.constraints.plot_tag,vehicle.constraints.engine.type,filename ='constraint_diagram_electric_liquid')

    electric_liquid_WS = constraint_results.des_wing_loading
    electric_liquid_TW = constraint_results.des_thrust_to_weight


    # expected values
    turboprop_WS_true       = 684.6558002
    turboprop_TW_true       = 8.5003630
    piston_WS_true          = 684.6558002
    piston_TW_true          = 8.500363024
    electric_air_TW_true    = 8.500363
    electric_air_WS_true    = 684.6558002
    electric_liquid_TW_true = 8.500363
    electric_liquid_WS_true = 684.6558002
    jet_WS_Matt_true        = 6412.7523310132865
    jet_TW_Matt_true        = 0.325505792
    jet_WS_Scholz_true      = 6412.7523310132865
    jet_TW_Scholz_true      = 0.25210340
    jet_WS_Howe_true        = 6412.7523310132865
    jet_TW_Howe_true        = 0.30980317
    jet_WS_Bartel_true      = 6412.7523310132865
    jet_TW_Bartel_true      = 0.31829093


    err_turboprop_WS        = (turboprop_WS - turboprop_WS_true)/turboprop_WS_true
    err_turboprop_TW        = (turboprop_TW - turboprop_TW_true)/turboprop_TW_true
    err_piston_WS           = (piston_WS - piston_WS_true)/piston_WS_true
    err_piston_TW           = (piston_TW - piston_TW_true)/piston_TW_true
    err_electric_air_TW     = (electric_air_TW - electric_air_TW_true)/electric_air_TW_true
    err_electric_air_WS     = (electric_air_WS - electric_air_WS_true)/electric_air_WS_true
    err_electric_liquid_TW  = (electric_liquid_TW - electric_liquid_TW_true)/electric_liquid_TW_true
    err_electric_liquid_WS  = (electric_liquid_TW - electric_liquid_TW_true)/electric_liquid_TW_true
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
    print('Piston              : W/S = ' + str(piston_WS) + ', P/W = ' + str(piston_TW))
    print('Electric air        : W/S = ' + str(electric_air_WS) + ', P/W = ' + str(electric_air_TW))
    print('Electric liquid     : W/S = ' + str(electric_liquid_WS) + ', P/W = ' + str(electric_liquid_TW))
    print('Turbofan, Mattingly : W/S = ' + str(jet_WS_Matt) + ', T/W = ' + str(jet_TW_Matt))    
    print('Turbofan, Scholz    : W/S = ' + str(jet_WS_Scholz) + ', T/W = ' + str(jet_TW_Scholz))   
    print('Turbofan, Howe      : W/S = ' + str(jet_WS_Howe) + ', T/W = ' + str(jet_TW_Howe))  
    print('Turbofan, Bartel    : W/S = ' + str(jet_WS_Bartel) + ', T/W = ' + str(jet_TW_Bartel)) 
    
    print('Expected values:')
    print('Turboprop           : W/S = ' + str(turboprop_WS_true)  + ', P/W = ' + str(turboprop_TW_true))
    print('Piston              : W/S = ' + str(piston_WS_true)     + ', P/W = ' + str(piston_TW_true))
    print('Electric air        : W/S = ' + str(electric_air_WS_true)   + ', P/W = ' + str(electric_air_TW_true))
    print('Electric liquid     : W/S = ' + str(electric_liquid_WS_true)   + ', P/W = ' + str(electric_liquid_TW_true))
    print('Turbofan, Mattingly : W/S = ' + str(jet_WS_Matt_true)   + ', T/W = ' + str(jet_TW_Matt_true))    
    print('Turbofan, Scholz    : W/S = ' + str(jet_WS_Scholz_true) + ', T/W = ' + str(jet_TW_Scholz_true))   
    print('Turbofan, Howe      : W/S = ' + str(jet_WS_Howe_true)   + ', T/W = ' + str(jet_TW_Howe_true))  
    print('Turbofan, Bartel    : W/S = ' + str(jet_WS_Bartel_true) + ', T/W = ' + str(jet_TW_Bartel_true)) 


    err = Data()
    err.turboprop_WS_error      = err_turboprop_WS
    err.turboprop_TW_error      = err_turboprop_TW
    err.piston_WS_error         = err_piston_WS
    err.piston_TW_error         = err_piston_TW
    err.electric_air_WS_error   = err_electric_air_WS
    err.electric_air_TW_error   = err_electric_air_TW
    err.electric_air_WS_error   = err_electric_liquid_WS
    err.electric_air_TW_error   = err_electric_liquid_TW

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
    print('Piston              : W/S = ' + str(err_piston_WS) + ', P/W = ' + str(err_piston_TW))
    print('Electric air        : W/S = ' + str(err_electric_air_WS) + ', P/W = ' + str(err_electric_air_TW))
    print('Electric liquid     : W/S = ' + str(err_electric_liquid_WS) + ', P/W = ' + str(err_electric_liquid_TW))
    print('Turbofan, Mattingly : W/S = ' + str(err_jet_WS_Matt) + ', T/W = ' + str(err_jet_TW_Matt))    
    print('Turbofan, Scholz    : W/S = ' + str(err_jet_WS_Scholz) + ', T/W = ' + str(err_jet_TW_Scholz))   
    print('Turbofan, Howe      : W/S = ' + str(err_jet_WS_Howe) + ', T/W = ' + str(err_jet_TW_Howe))  
    print('Turbofan, Bartel    : W/S = ' + str(err_jet_WS_Bartel) + ', T/W = ' + str(err_jet_TW_Bartel)) 



    for k,v in list(err.items()):
        assert(np.abs(v)<1E-6)    

    
if __name__ == '__main__':
    main()




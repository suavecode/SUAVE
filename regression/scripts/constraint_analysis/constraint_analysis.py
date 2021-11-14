#constraint_analysis.py

# Created:  Dec 2020, S. Karpuk
# Modified: 


#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
import SUAVE.Analyses.Constraint_Analysis as Constraint_Analysis
import numpy as np
import matplotlib.pyplot as plt

from SUAVE.Core import Data, Units


def main():

    # Sample jet airplane
    # ------------------------------------------------------------------
    ca = Constraint_Analysis.Constraint_Analysis()

    ca.plot_units = 'US'

    # Define default constraint analysis
    ca.analyses.takeoff    = True
    ca.analyses.cruise     = True
    ca.analyses.max_cruise = False
    ca.analyses.landing    = True
    ca.analyses.OEI_climb  = True
    ca.analyses.turn       = True
    ca.analyses.climb      = True
    ca.analyses.ceiling    = True

    # take-off
    ca.takeoff.runway_elevation = 0 * Units['meter']
    ca.takeoff.ground_run       = 1550 * Units['m']
    # climb
    ca.climb.altitude   = 35000 * Units['feet']
    ca.climb.airspeed   = 450   * Units['knots']
    ca.climb.climb_rate = 1.7   * Units['m/s']
    #cruise
    ca.cruise.altitude        = 35000 * Units['feet']  
    ca.cruise.mach            = 0.78
    ca.cruise.thrust_fraction = 0.85
    # turn
    ca.turn.angle           = 15    * Units['degrees']
    ca.turn.altitude        = 35000 * Units['feet']   
    ca.turn.mach            = 0.78
    ca.turn.thrust_fraction = 0.85
    # ceiling
    ca.ceiling.altitude    = 36500 * Units['feet'] 
    ca.ceiling.mach            = 0.78
    # landing
    ca.landing.ground_roll = 1400 * Units['m']   

    # Default aircraft properties
    # geometry
    ca.geometry.aspect_ratio           = 9.16
    ca.geometry.thickness_to_chord     = 0.11
    ca.geometry.taper                  = 0.3
    ca.geometry.sweep_quarter_chord    = 25 * Units['degrees']
    ca.geometry.high_lift_type_clean   = None
    ca.geometry.high_lift_type_takeoff = 'double-slotted Fowler'
    ca.geometry.high_lift_type_landing = 'double-slotted Fowler'
    # engine
    ca.engine.type         = 'turbofan'
    ca.engine.number       = 2
    ca.engine.bypass_ratio = 6.0
    

    # Define aerodynamics
    ca.aerodynamics.cd_takeoff     = 0.044
    ca.aerodynamics.cl_takeoff     = 0.6
    ca.aerodynamics.cl_max_clean   = 1.35
    ca.aerodynamics.cd_min_clean   = 0.0134
    ca.aerodynamics.cl_max_takeoff = 2.1
    ca.aerodynamics.cl_max_landing = 2.4

    ca.design_point_type = 'maximum wing loading'

    # run the constraint diagrams for various engine models

    ca.engine.method = "Mattingly"
    ca.create_constraint_diagram()

    jet_WS_Matt = ca.des_wing_loading
    jet_TW_Matt = ca.des_thrust_to_weight

    ca.engine.method = "Scholz"
    ca.create_constraint_diagram()

    jet_WS_Scholz = ca.des_wing_loading
    jet_TW_Scholz = ca.des_thrust_to_weight

    ca.engine.method = "Howe"
    ca.create_constraint_diagram()

    jet_WS_Howe = ca.des_wing_loading
    jet_TW_Howe = ca.des_thrust_to_weight
    
    ca.engine.method = "Bartel"
    ca.create_constraint_diagram()

    jet_WS_Bartel = ca.des_wing_loading
    jet_TW_Bartel = ca.des_thrust_to_weight

 

    # Sample propeller airplanes
    # ------------------------------------------------------------------
    ca = Constraint_Analysis.Constraint_Analysis()

    ca.plot_units = 'US'

    # Define default constraint analysis
    ca.analyses.takeoff     = True
    ca.analyses.cruise      = True
    ca.analyses.max_cruise  = False
    ca.analyses.landing     = True
    ca.analyses.OEI_climb   = True
    ca.analyses.turn        = True
    ca.analyses.climb       = True
    ca.analyses.ceiling     = True

    # take-off
    ca.takeoff.runway_elevation = 0 * Units['meter']
    ca.takeoff.ground_run       = 1250 * Units['m']
    # climb
    ca.climb.altitude   = 20000 * Units['feet']
    ca.climb.climb_rate = 4   * Units['m/s']
    #cruise
    ca.cruise.altitude        = 20000 * Units['feet']  
    ca.cruise.mach            = 0.42
    ca.cruise.thrust_fraction = 1
    # turn
    ca.turn.angle           = 15    * Units['degrees']
    ca.turn.altitude        = 20000 * Units['feet']   
    ca.turn.mach            = 0.42
    ca.turn.thrust_fraction = 1
    # ceiling
    ca.ceiling.altitude    = 25000 * Units['feet'] 
    ca.ceiling.mach            = 0.42
    # landing
    ca.landing.ground_roll = 650 * Units['m']   

    # Default aircraft properties
    # geometry
    ca.geometry.aspect_ratio           = 12.0
    ca.geometry.thickness_to_chord     = 0.15
    ca.geometry.taper                  = 0.5
    ca.geometry.sweep_quarter_chord    = 0.0 * Units['degrees']
    ca.geometry.high_lift_type_clean   = None

    # engine
    ca.engine.number = 2
    # propeller
    ca.propeller.takeoff_efficiency   = 0.5
    ca.propeller.climb_efficiency     = 0.8
    ca.propeller.cruise_efficiency    = 0.85
    ca.propeller.turn_efficiency      = 0.85
    ca.propeller.ceiling_efficiency   = 0.85
    ca.propeller.OEI_climb_efficiency = 0.5

    # Define aerodynamics
    ca.aerodynamics.cd_takeoff     = 0.04
    ca.aerodynamics.cl_takeoff     = 0.6
    ca.aerodynamics.cl_max_clean   = 1.35
    ca.aerodynamics.cd_min_clean   = 0.02

    ca.design_point_type = 'minimum power-to-weight'

    # run the constraint diagram for various engine ad flap types
    ca.engine.type = 'turboprop'
    ca.geometry.high_lift_type_takeoff = 'single-slotted'
    ca.geometry.high_lift_type_landing = 'single-slotted'
    ca.create_constraint_diagram()

    turboprop_WS = ca.des_wing_loading
    turboprop_TW = ca.des_thrust_to_weight

    ca.engine.type = 'piston'
    ca.geometry.high_lift_type_takeoff = 'plain'
    ca.geometry.high_lift_type_landing = 'plain'
    ca.create_constraint_diagram()

    piston_WS = ca.des_wing_loading
    piston_TW = ca.des_thrust_to_weight

    ca.engine.type = 'electric'
    ca.geometry.high_lift_type_takeoff = 'double-slotted fixed vane'
    ca.geometry.high_lift_type_landing = 'double-slotted fixed vane'
    ca.create_constraint_diagram()

    electric_WS = ca.des_wing_loading
    electric_TW = ca.des_thrust_to_weight


    # expected values
    turboprop_WS_true       = 219.7047819
    turboprop_TW_true       = 0.164536375
    piston_WS_true          = 219.7047819
    piston_TW_true          = 0.221022812
    electric_WS_true        = 195.2931395
    electric_TW_true        = 0.153919647
    jet_WS_Matt_true        = 653.905472981
    jet_TW_Matt_true        = 3.3077663960
    jet_WS_Scholz_true      = 653.905472981
    jet_TW_Scholz_true      = 2.962242635
    jet_WS_Howe_true        = 653.905472981
    jet_TW_Howe_true        = 3.0046653126
    jet_WS_Bartel_true      = 653.905472981
    jet_TW_Bartel_true      = 3.89728490


    err_turboprop_WS    = (turboprop_WS - turboprop_WS_true)/turboprop_WS_true
    err_turboprop_TW    = (turboprop_TW - turboprop_TW_true)/turboprop_TW_true
    err_piston_WS       = (piston_WS - piston_WS_true)/piston_WS_true
    err_piston_TW       = (piston_TW - piston_TW_true)/piston_TW_true
    err_electric_WS     = (electric_WS - electric_WS_true)/electric_WS_true
    err_electric_TW     = (electric_TW - electric_TW_true)/electric_TW_true
    err_jet_WS_Matt     = (jet_WS_Matt  - jet_WS_Matt_true)/jet_WS_Matt_true
    err_jet_TW_Matt     = (jet_TW_Matt  - jet_TW_Matt_true)/jet_TW_Matt_true
    err_jet_WS_Scholz   = (jet_WS_Scholz  - jet_WS_Scholz_true)/jet_WS_Scholz_true
    err_jet_TW_Scholz   = (jet_TW_Scholz  - jet_TW_Scholz_true)/jet_TW_Scholz_true
    err_jet_WS_Howe     = (jet_WS_Howe  - jet_WS_Howe_true)/jet_WS_Howe_true
    err_jet_TW_Howe     = (jet_TW_Howe - jet_TW_Howe_true)/jet_TW_Howe_true
    err_jet_WS_Bartel   = (jet_WS_Bartel  - jet_WS_Bartel_true)/jet_WS_Bartel_true
    err_jet_TW_Bartel   = (jet_TW_Bartel - jet_TW_Bartel_true)/jet_TW_Bartel_true
 

    print('Calculated values:')
    print('Turboprop           : W/S = ' + str(turboprop_WS) + ', P/W = ' + str(turboprop_TW))
    print('Piston              : W/S = ' + str(piston_WS) + ', P/W = ' + str(piston_TW))
    print('Electric            : W/S = ' + str(electric_WS) + ', P/W = ' + str(electric_TW))
    print('Turbofan, Mattingly : W/S = ' + str(jet_WS_Matt) + ', T/W = ' + str(jet_TW_Matt))    
    print('Turbofan, Scholz    : W/S = ' + str(jet_WS_Scholz) + ', T/W = ' + str(jet_TW_Scholz))   
    print('Turbofan, Howe      : W/S = ' + str(jet_WS_Howe) + ', T/W = ' + str(jet_TW_Howe))  
    print('Turbofan, Bartel    : W/S = ' + str(jet_WS_Bartel) + ', T/W = ' + str(jet_TW_Bartel)) 
    
    print('Expected values:')
    print('Turboprop           : W/S = ' + str(turboprop_WS_true)  + ', P/W = ' + str(turboprop_TW_true))
    print('Piston              : W/S = ' + str(piston_WS_true)     + ', P/W = ' + str(piston_TW_true))
    print('Electric            : W/S = ' + str(electric_WS_true)   + ', P/W = ' + str(electric_TW_true))
    print('Turbofan, Mattingly : W/S = ' + str(jet_WS_Matt_true)   + ', T/W = ' + str(jet_TW_Matt_true))    
    print('Turbofan, Scholz    : W/S = ' + str(jet_WS_Scholz_true) + ', T/W = ' + str(jet_TW_Scholz_true))   
    print('Turbofan, Howe      : W/S = ' + str(jet_WS_Howe_true)   + ', T/W = ' + str(jet_TW_Howe_true))  
    print('Turbofan, Bartel    : W/S = ' + str(jet_WS_Bartel_true) + ', T/W = ' + str(jet_TW_Bartel_true)) 


    err = Data()
    err.turboprop_WS_error      = err_turboprop_WS
    err.turboprop_TW_error      = err_turboprop_TW
    err.piston_WS_error         = err_piston_WS
    err.piston_TW_error         = err_piston_TW
    err.electric_WS_error       = err_electric_WS
    err.electri_TW_error        = err_electric_TW

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
    print('Electric            : W/S = ' + str(err_electric_WS) + ', P/W = ' + str(err_electric_TW))
    print('Turbofan, Mattingly : W/S = ' + str(err_jet_WS_Matt) + ', T/W = ' + str(err_jet_TW_Matt))    
    print('Turbofan, Scholz    : W/S = ' + str(err_jet_WS_Scholz) + ', T/W = ' + str(err_jet_TW_Scholz))   
    print('Turbofan, Howe      : W/S = ' + str(err_jet_WS_Howe) + ', T/W = ' + str(err_jet_TW_Howe))  
    print('Turbofan, Bartel    : W/S = ' + str(err_jet_WS_Bartel) + ', T/W = ' + str(err_jet_TW_Bartel)) 



    for k,v in list(err.items()):
        assert(np.abs(v)<1E-6)    

    
if __name__ == '__main__':
    main()




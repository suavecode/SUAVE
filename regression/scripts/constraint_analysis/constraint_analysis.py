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

from Boeing_737 import vehicle_setup as  B737_vehicle
from Cessna_172 import vehicle_setup as  Cessna172_vehicle


def main():

    # Sample jet airplane
    # ------------------------------------------------------------------
    ca = Constraint_Analysis.Constraint_Analysis()

    # Define default constraint analysis
    ca.analyses.takeoff    = True
    ca.analyses.cruise     = True
    ca.analyses.max_cruise = False
    ca.analyses.landing    = True
    ca.analyses.OEI_climb  = True
    ca.analyses.turn       = False
    ca.analyses.climb      = True
    ca.analyses.ceiling    = False

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
    # landing
    ca.landing.ground_roll = 1400 * Units['m']  

    # define vehicle-specific properties
    vehicle  = B737_vehicle()


    # Default aircraft properties
    # geometry
    ca.geometry.aspect_ratio                 = vehicle.wings['main_wing'].aspect_ratio
    ca.geometry.thickness_to_chord           = vehicle.wings['main_wing'].thickness_to_chord
    ca.geometry.taper                        = vehicle.wings['main_wing'].taper
    ca.geometry.sweep_quarter_chord          = vehicle.wings['main_wing'].sweeps.quarter_chord
    ca.geometry.high_lift_configuration_type = vehicle.wings['main_wing'].control_surfaces.flap.configuration_type 
    # engine
    ca.engine.type         = 'turbofan'
    ca.engine.number       = vehicle.networks.turbofan.number_of_engines
    ca.engine.bypass_ratio = vehicle.networks.turbofan.bypass_ratio
    
    # Define aerodynamics (an example with pre-defined max lift coefficients)
    ca.aerodynamics.cd_takeoff     = 0.044
    ca.aerodynamics.cl_takeoff     = 0.6
    ca.aerodynamics.cd_min_clean   = 0.0134
    ca.aerodynamics.cl_max_takeoff = 2.1
    ca.aerodynamics.cl_max_landing = 2.4

    ca.design_point_type = 'maximum wing loading'

    # run the constraint diagrams for various engine models

    ca.engine.method = "Mattingly"
    ca.create_constraint_diagram()
    plot_constraint_diagram(ca)

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
    ca.takeoff.ground_run       = 450 * Units['m']
    # climb
    ca.climb.altitude   = 0.0   * Units['meter']
    ca.climb.climb_rate = 3.66   * Units['m/s']
    #cruise
    ca.cruise.altitude        = 4000 * Units['meter']  
    ca.cruise.mach            = 0.2
    ca.cruise.thrust_fraction = 1
    # turn
    ca.turn.angle           = 15    * Units['degrees']
    ca.turn.altitude        = 3000  * Units['meter']   
    ca.turn.mach            = 0.2
    ca.turn.thrust_fraction = 1
    # ceiling
    ca.ceiling.altitude     = 4000 * Units['meter'] 
    ca.ceiling.mach         = 0.2
    # landing
    ca.landing.ground_roll = 120 * Units['m']   

    # define vehicle-specific properties
    vehicle  = Cessna172_vehicle()

    ca.analyses.OEI_climb   = False

    # Default aircraft properties
    # geometry
    ca.geometry.aspect_ratio                    = vehicle.wings['main_wing'].aspect_ratio
    ca.geometry.thickness_to_chord              = vehicle.wings['main_wing'].thickness_to_chord
    ca.geometry.taper                           = vehicle.wings['main_wing'].taper
    ca.geometry.sweep_quarter_chord             = vehicle.wings['main_wing'].sweeps.quarter_chord
    ca.geometry.high_lift_configuration_type    = vehicle.wings['main_wing'].control_surfaces.flap.configuration_type

    # engine
    ca.engine.number = vehicle.networks.internal_combustion.number_of_engines
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

    # run the constraint diagram for various engine ad flap types
    ca.engine.type = 'turboprop'
    ca.geometry.high_lift_configuration_type = 'single-slotted'
    ca.create_constraint_diagram()

    turboprop_WS = ca.des_wing_loading
    turboprop_TW = ca.des_thrust_to_weight

    ca.engine.type = 'piston'
    ca.geometry.high_lift_configuration_type = 'plain'
    ca.create_constraint_diagram()
    plot_constraint_diagram(ca,filename ='constraint_diagram_piston')

    piston_WS = ca.des_wing_loading
    piston_TW = ca.des_thrust_to_weight

    ca.engine.type = 'electric air-cooled'
    ca.geometry.high_lift_configuration_type = 'double-slotted fixed vane'
    ca.create_constraint_diagram()
    plot_constraint_diagram(ca,filename ='constraint_diagram_electric_air')

    electric_air_WS = ca.des_wing_loading
    electric_air_TW = ca.des_thrust_to_weight

    ca.engine.type = 'electric liquid-cooled'
    ca.geometry.high_lift_configuration_type = 'double-slotted fixed vane'
    ca.create_constraint_diagram()
    plot_constraint_diagram(ca,filename ='constraint_diagram_electric_liquid')

    electric_liquid_WS = ca.des_wing_loading
    electric_liquid_TW = ca.des_thrust_to_weight


    # expected values
    turboprop_WS_true       = 629.027516462
    turboprop_TW_true       = 8.924349473
    piston_WS_true          = 516.3445826
    piston_TW_true          = 10.9441325
    electric_air_TW_true    = 7.1642177
    electric_air_WS_true    = 780.22233
    electric_liquid_TW_true = 7.1642177
    electric_liquid_WS_true = 502.742719
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




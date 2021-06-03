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
    ca = Constraint_Analysis.Constraint_Analysis()

    ca.plot_units = 'US'

    # Define default constraint analysis
    ca.analyses.takeoff   = True
    ca.analyses.cruise    = True
    ca.analyses.max_cruise = False
    ca.analyses.landing   = True
    ca.analyses.OEI_climb = True
    ca.analyses.turn      = True
    ca.analyses.climb     = True
    ca.analyses.ceiling   = True

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
    ca.engine.method       = "Mattingly"

    # Define aerodynamics
    ca.aerodynamics.cd_takeoff     = 0.044
    ca.aerodynamics.cl_takeoff     = 0.6
    #ca.aerodynamics.cl_max_takeoff = 2.0
    #ca.aerodynamics.cl_max_landing = 2.2
    ca.aerodynamics.cl_max_clean   = 1.35
    ca.aerodynamics.cd_min_clean   = 0.0134

    ca.design_point_type = 'maximum wing loading'

    # run the constraint diagram
    ca.create_constraint_diagram()

    jet_WS = ca.des_wing_loading
    jet_TW = ca.des_thrust_to_weight


    # Sample propeller airplane
    ca = Constraint_Analysis.Constraint_Analysis()

    ca.plot_units = 'US'

    # Define default constraint analysis
    ca.analyses.takeoff   = True
    ca.analyses.cruise    = True
    ca.analyses.max_cruise = False
    ca.analyses.landing   = True
    ca.analyses.OEI_climb = True
    ca.analyses.turn      = True
    ca.analyses.climb     = True
    ca.analyses.ceiling   = True

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
    ca.geometry.high_lift_type_takeoff = 'single-slotted Fowler'
    ca.geometry.high_lift_type_landing = 'single-slotted Fowler'
    # engine
    ca.engine.type         = 'turboprop'
    ca.engine.number       = 2
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
    ca.aerodynamics.cl_max_takeoff = 2.3
    #ca.aerodynamics.cl_max_landing = 2.2
    ca.aerodynamics.cl_max_clean   = 1.35
    ca.aerodynamics.cd_min_clean   = 0.02

    ca.design_point_type = 'minimum power-to-weight'

    # run the constraint diagram
    ca.create_constraint_diagram()

    prop_WS = ca.des_wing_loading
    prop_TW = ca.des_thrust_to_weight

    # true values
    prop_WS_truth = 244.116424
    prop_TW_truth = 0.1791623
    jet_WS_truth  = 725.143706
    jet_TW_truth  = 3.694133

    err_prop_WS = (prop_WS - prop_WS_truth)/prop_WS_truth
    err_prop_TW = (prop_TW - prop_TW_truth)/prop_TW_truth 
    err_jet_WS  = (jet_WS  - jet_WS_truth)/jet_WS_truth
    err_jet_TW  = (jet_TW  - jet_TW_truth)/jet_TW_truth
    
    err       = Data()
    err.propeller_WS_error = err_prop_WS
    err.propeller_TW_error = err_prop_TW
    err.jet_WS_error       = err_jet_WS
    err.jet_TW_error       = err_jet_TW

    for k,v in list(err.items()):
        assert(np.abs(v)<1E-6)    

    
if __name__ == '__main__':
    main()

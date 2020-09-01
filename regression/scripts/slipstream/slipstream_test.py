# Slipstream_Test.py
# 
# Created:  Mar 2019, M. Clarke

""" setup file for a cruise segment of the NASA X-57 Maxwell (Twin Engine Variant) Electric Aircraft 
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt

from SUAVE.Core import Data , Container
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Plots.Mission_Plots import *  
import sys
sys.path.append('../Vehicles') 
from X57_Maxwell import vehicle_setup, configs_setup 

import copy

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():

    configs, analyses = full_setup() 
    
    configs.finalize()
    analyses.finalize()  

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
     
    # lift coefficient  
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[1][0]
    lift_coefficient_true         = 0.4724469261316692

    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([1.29656806e-01, 1.27033015e-01, 1.17459818e-01, 8.77349363e-02,
                                                1.01534160e-01, 1.32358366e-01, 9.85520151e-02, 8.25103473e-02,
                                                6.88810155e-02, 5.52314102e-02, 4.17770387e-02, 2.89819366e-02,
                                                1.74023495e-02, 7.80713375e-03, 1.55901625e-03, 1.29718112e-01,
                                                1.27198008e-01, 1.17676369e-01, 8.78848736e-02, 1.01788681e-01,
                                                1.32744718e-01, 9.87903226e-02, 8.26827384e-02, 6.90085144e-02,
                                                5.53210138e-02, 4.18351397e-02, 2.90152480e-02, 1.74179723e-02,
                                                7.81213671e-03, 1.55961702e-03, 2.63471097e-03, 2.78345074e-03,
                                                2.95720984e-03, 3.14215407e-03, 3.31239311e-03, 3.44043935e-03,
                                                3.49541544e-03, 3.43964418e-03, 3.23946475e-03, 2.87784399e-03,
                                                2.36205414e-03, 1.73144869e-03, 1.06487648e-03, 4.78294353e-04,
                                                9.56061003e-05, 2.63555062e-03, 2.78601227e-03, 2.96130063e-03,
                                                3.14746881e-03, 3.31861181e-03, 3.44724269e-03, 3.50246134e-03,
                                                3.44652999e-03, 3.24574046e-03, 2.88308589e-03, 2.36596427e-03,
                                                1.73394405e-03, 1.06613146e-03, 4.78707481e-04, 9.56553172e-05,
                                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                                0.00000000e+00, 0.00000000e+00, 0.00000000e+00])

    
    print(sectional_lift_coeff)
    diff_Cl                       = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    assert  max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-6 

    # plot results 
    plot_mission(results,configs.base)     
    return 

def plot_mission(results,vehicle): 
    
    # Plot surface pressure coefficient 
    plot_surface_pressure_contours(results,vehicle)
    
    # Plot lift distribution 
    plot_lift_distribution(results,vehicle)
    
    # Create Video Frames 
    create_video_frames(results,vehicle, save_figure = False)
    
    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup():

    # vehicle data
    vehicle  = vehicle_setup() 
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle) 
    missions_analyses = missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
        analyses[tag] = analysis

    return analyses

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()     
    aerodynamics.settings.use_surrogate              = False
    aerodynamics.settings.prop_wake_model            = True 
    aerodynamics.settings.number_panels_spanwise     = 15
    aerodynamics.settings.number_panels_chordwise    = 5   
    aerodynamics.geometry                            = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Stability Analysis
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()    
    stability.geometry = vehicle
    analyses.append(stability)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors 
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   

    # done!
    return analyses    


# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses,vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0. * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments 
    
    # base segment
    base_segment = Segments.Segment()
    ones_row     = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 2
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.battery_propeller.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.battery_propeller.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.2 * ones_row(1) 
    bat                                                      = vehicle.propulsors.battery_propeller.battery 
    base_segment.state.unknowns.battery_voltage_under_load   = bat.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2) 
    base_segment.max_energy                                  = bat.max_energy 
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 12000  * Units.feet
    segment.air_speed                 = 135.   * Units['mph'] 
    segment.distance                  =  20.   * Units.nautical_mile  
    segment.state.unknowns.throttle   = 0.85  *  ones_row(1)    
    segment.battery_energy            = bat.max_energy   
    
    # add to misison
    mission.append_segment(segment)    
    
    return mission



def missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------

    missions.base = base_mission

    # done!
    return missions  


if __name__ == '__main__': 
    main()    
    plt.show()
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
    lift_coefficient_true         = 0.4725344379782014

    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([1.27520594e-01, 1.24876208e-01, 1.15544474e-01, 8.56027093e-02,
                                                9.98143764e-02, 1.31570444e-01, 9.67708543e-02, 8.09092096e-02,
                                                6.74356187e-02, 5.40115991e-02, 4.08216076e-02, 2.83030692e-02,
                                                1.69883534e-02, 7.61988113e-03, 1.52167045e-03, 1.27672829e-01,
                                                1.25647377e-01, 1.18076497e-01, 8.38100889e-02, 1.23510185e-01,
                                                1.39101666e-01, 1.01792299e-01, 8.27698438e-02, 6.85203148e-02,
                                                5.47028922e-02, 4.12636069e-02, 2.85727042e-02, 1.71350733e-02,
                                                7.68110111e-03, 1.53329132e-03, 2.51672785e-03, 2.61535233e-03,
                                                2.75391050e-03, 2.91502890e-03, 3.07152641e-03, 3.19545896e-03,
                                                3.25600448e-03, 3.21553761e-03, 3.03974234e-03, 2.70992083e-03,
                                                2.23090667e-03, 1.63909814e-03, 1.00968063e-03, 4.53923457e-04,
                                                9.07650352e-05, 2.60463336e-03, 2.79811035e-03, 3.00027659e-03,
                                                3.20071103e-03, 3.37617686e-03, 3.50214248e-03, 3.55156758e-03,
                                                3.49054566e-03, 3.28640678e-03, 2.92001705e-03, 2.39647237e-03,
                                                1.75548040e-03, 1.07842348e-03, 4.83809833e-04, 9.66356072e-05,
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
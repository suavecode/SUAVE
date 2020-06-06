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
    lift_coefficient_true         = 0.472549171541998
    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6
    
    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([1.30731200e-01, 1.29518975e-01, 1.25476612e-01, 1.19200829e-01,
                                                1.11094027e-01, 1.01475358e-01, 9.06194145e-02, 7.87825359e-02,
                                                6.62329301e-02, 5.32930082e-02, 4.03869121e-02, 2.80468082e-02,
                                                1.68504855e-02, 7.56183132e-03, 1.51037470e-03, 1.30793344e-01,
                                                1.29689486e-01, 1.25716810e-01, 1.19468403e-01, 1.11355370e-01,
                                                1.01709356e-01, 9.08158736e-02, 7.89387974e-02, 6.63507579e-02,
                                                5.33764629e-02, 4.04412349e-02, 2.80780177e-02, 1.68651393e-02,
                                                7.56652692e-03, 1.51093870e-03, 2.27948231e-03, 2.38250557e-03,
                                                2.47826742e-03, 2.55290540e-03, 2.58516086e-03, 2.55807996e-03,
                                                2.46129251e-03, 2.28797933e-03, 2.03784971e-03, 1.71999532e-03,
                                                1.35169778e-03, 9.58096403e-04, 5.75716849e-04, 2.55087240e-04,
                                                5.07367423e-05, 2.28011348e-03, 2.38439254e-03, 2.48115957e-03,
                                                2.55644055e-03, 2.58898057e-03, 2.56187560e-03, 2.46482121e-03,
                                                2.29105992e-03, 2.04036324e-03, 1.72189076e-03, 1.35299172e-03,
                                                9.58865123e-04, 5.76083617e-04, 2.55204075e-04, 5.07504917e-05,
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
    aerodynamics.settings.integrate_slipstream       = True 
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
# Slipstream_Test.py
#
# Created:  Mar 2019, M. Clarke
# Modified: Jun 2021, R. Erhard
#           Feb 2022, R. Erhard

""" setup file for a cruise segment of the NASA X-57 Maxwell (Twin Engine Variant) Electric Aircraft
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt
import sys

from SUAVE.Plots.Performance.Mission_Plots import *
from SUAVE.Plots.Geometry.plot_vehicle import plot_vehicle
from SUAVE.Plots.Geometry.plot_vehicle_vlm_panelization  import plot_vehicle_vlm_panelization

from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One
sys.path.append('../Vehicles')
from X57_Maxwell_Mod2 import vehicle_setup, configs_setup

import time

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    #run test with helical fixed wake model
    t0 = time.time()
    Fidelity_One_Wake_Analysis(identical_props=True)
    print("TIME: " + str(time.time()-t0))

    # run test with helical fixed wake model and non-identical props    
    t0 = time.time()
    Fidelity_One_Wake_Analysis(identical_props=False)
    print("TIME: " + str((time.time()-t0)/60))   
    
    # run test with bevw wake model
    Fidelity_Zero_Wake_Analysis()

    return

def Fidelity_Zero_Wake_Analysis():
    # Evaluate wing in propeller wake (using Fidelity Zero wake model)
    wake_fidelity = 0
    configs, analyses  = full_setup(wake_fidelity, identical_props=True)

    configs.finalize()
    analyses.finalize()

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # lift coefficient
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[1][0]
    lift_coefficient_true         = 0.43768245404502637


    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true)
    print('CL difference')
    print(diff_CL)


    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6

    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([ 4.57933495e-01,  3.31030791e-01,  3.72322963e-01,  3.25750054e-01,
                                                 6.30279073e-02,  4.57933511e-01,  3.31030802e-01,  3.72323014e-01,
                                                 3.25750268e-01,  6.30279328e-02, -5.76708908e-02, -5.54647407e-02,
                                                -4.78313483e-02, -3.37720002e-02, -1.91780281e-02, -5.76709064e-02,
                                                -5.54647633e-02, -4.78313797e-02, -3.37720401e-02, -1.91780399e-02,
                                                -2.12104101e-15, -8.27709312e-16, -6.08914689e-16, -4.63674304e-16,
                                                -2.94742692e-16])



    print(sectional_lift_coeff)
    diff_Cl   = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    assert  np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-6

    # plot results
    plot_mission(results,configs.base)

    # Plot vehicle
    plot_vehicle(configs.base, save_figure = False, plot_control_points = False)

    # Plot vortex distribution
    plot_vehicle_vlm_panelization(configs.base, save_figure=False, plot_control_points=True)
    return

def Fidelity_One_Wake_Analysis(identical_props):
    # Evaluate wing in propeller wake (using helical fixed-wake model)
    wake_fidelity = 1
    configs, analyses = full_setup(wake_fidelity, identical_props)

    configs.finalize()
    analyses.finalize()

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # lift coefficient
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[1][0]
    lift_coefficient_true         = 0.43813579094153554

    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true)
    print('CL difference')
    print(diff_CL)


    # sectional lift coefficient check
    sectional_lift_coeff            = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    sectional_lift_coeff_true       = np.array([ 4.17980236e-01,  6.35637114e-01,  3.48935801e-01,  2.65231869e-01,
                                                 5.06858059e-02,  3.89937920e-01,  4.10040539e-01,  2.94170309e-01,
                                                 2.40561453e-01,  4.58450189e-02, -1.14517416e-01, -1.10479319e-01,
                                                -9.96857378e-02, -8.28741705e-02, -5.29456124e-02, -9.38371058e-02,
                                                -8.94528129e-02, -8.26897371e-02, -6.62645032e-02, -4.09864507e-02,
                                                -6.88346479e-07, -8.74875486e-10, -1.47356561e-08, -2.68471275e-08,
                                                -1.54810932e-08])


    plot_lift_distribution(results,configs.base)
    
    print(sectional_lift_coeff)
    diff_Cl                       = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    

    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6    
    assert  np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-6

    # plot results
    plot_mission(results,configs.base)

    # Plot vehicle
    plot_vehicle(configs.base, save_figure = False, plot_control_points = False)

    # Plot vortex distribution
    plot_vehicle_vlm_panelization(configs.base, save_figure=False, plot_control_points=True)
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

def full_setup(wake_fidelity, identical_props):

    # vehicle data
    vehicle  = vehicle_setup()
    # update wake method and rotation direction of rotors:
    props = vehicle.networks.battery_propeller.propellers
    for p in props:
        p.rotation = -1
        if wake_fidelity==1:
            p.Wake = Rotor_Wake_Fidelity_One()   
            p.Wake.wake_settings.number_rotor_rotations = 1
            

    # test for non-identical propellers
    if not identical_props:
        vehicle.networks.battery_propeller.identical_propellers = False
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
    aerodynamics.settings.propeller_wake_model       = True

    aerodynamics.settings.number_spanwise_vortices   = 5
    aerodynamics.settings.number_chordwise_vortices  = 2
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
    energy.network = vehicle.networks
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

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    segment.analyses.extend(analyses.base)
    segment.battery_energy            = vehicle.networks.battery_propeller.battery.max_energy* 0.7
    segment.altitude                  = 8012 * Units.feet
    segment.air_speed                 = 135. * Units['mph']
    segment.distance                  = 20.  * Units.nautical_mile
    segment.state.unknowns.throttle   = 0.85 * ones_row(1)

    # post-process aerodynamic derivatives in cruise
    segment.process.finalize.post_process.aero_derivatives = SUAVE.Methods.Flight_Dynamics.Static_Stability.compute_aero_derivatives

    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)

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

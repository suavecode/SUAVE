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
from SUAVE.Core import Units, Data

import numpy as np
import pylab as plt
import sys

from SUAVE.Plots.Performance.Aerodynamics import *
from SUAVE.Plots.Geometry.Three_Dimensional.plot_3d_vehicle                            import plot_3d_vehicle
from SUAVE.Plots.Geometry.Three_Dimensional.plot_3d_vehicle_vlm_panelization           import plot_3d_vehicle_vlm_panelization  

from SUAVE.Analyses.Propulsion.Rotor_Wake_Fidelity_One import Rotor_Wake_Fidelity_One
from SUAVE.Methods.Aerodynamics.Common.Fidelity_Zero.Lift.VLM import  VLM  
from SUAVE.Analyses.Aerodynamics import Vortex_Lattice


sys.path.append('../Vehicles')
from X57_Maxwell_Mod2 import vehicle_setup, configs_setup
from Stopped_Rotor import vehicle_setup as V2
from Stopped_Rotor import configs_setup as configs2

import time

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    
    
    
    # fidelity zero wakes
    print('Wake Fidelity Zero, Identical Props')    
    t0=time.time()
    Propeller_Slipstream(wake_fidelity=0,identical_props=True)
    print((time.time()-t0)/60)
    
    # fidelity one wakes
    print('Wake Fidelity One, Identical Props')  
    t0=time.time()
    Propeller_Slipstream(wake_fidelity=1,identical_props=True)  
    print((time.time()-t0)/60)
    

    print('Wake Fidelity One, Non-Identical Props')      
    t0=time.time()
    Propeller_Slipstream(wake_fidelity=1,identical_props=False)  
    print((time.time()-t0)/60)
    
    return


def Propeller_Slipstream(wake_fidelity,identical_props):
    # setup configs, analyses
    configs, analyses  = X57_setup(wake_fidelity=wake_fidelity, identical_props=identical_props)
    
    # finalize configs
    configs.finalize()
    analyses.finalize()

    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
    
    # check regression values
    if wake_fidelity==0: 
        regress_1a(results,configs)
    elif wake_fidelity==1: 
        regress_1b(results, configs)
    
    return

def regress_1a(results, configs):
    # Regression for Stopped Rotor Test (using Fidelity Zero wake model)
    lift_coefficient            = results.segments.cruise.conditions.aerodynamics.lift_coefficient[1][0]
    sectional_lift_coeff        = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    
    # lift coefficient and sectional lift coefficient check
    lift_coefficient_true       = 0.6021342379985749
    sectional_lift_coeff_true   = np.array([ 5.77234192e-01,  5.23424701e-01,  4.85117608e-01,  4.19066081e-01,
                                             8.17527532e-02,  5.77234189e-01,  5.23424707e-01,  4.85117732e-01,
                                             4.19066382e-01,  8.17527978e-02,  3.46155791e-03,  1.86978089e-03,
                                             5.24101436e-04,  6.01240333e-04,  6.29838507e-04,  3.46156312e-03,
                                             1.86978968e-03,  5.24113979e-04,  6.01256772e-04,  6.29846414e-04,
                                             3.70758346e-16, -3.20030721e-16, -3.68491317e-16, -3.00336901e-16,
                                            -1.90343233e-16])

    diff_CL = np.abs(lift_coefficient  - lift_coefficient_true)
    print('CL difference')
    print(diff_CL)

    diff_Cl   = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)
    print('Cl difference')
    print(diff_Cl)
    
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6
    assert  np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-6

    # plot results, vehicle, and vortex distribution
    plot_mission(results,configs.base)
    #plot_3d_vehicle(configs.base, save_figure = False, plot_wing_control_points = False)
    #plot_3d_vehicle_vlm_panelization(configs.base, save_figure=False, plot_wing_control_points=True)
              
    return

def regress_1b(results, configs):
    # Regression for Stopped Rotor Test (using Fidelity One wake model)
    lift_coefficient            = results.segments.cruise.conditions.aerodynamics.lift_coefficient[1][0]
    sectional_lift_coeff        = results.segments.cruise.conditions.aerodynamics.lift_breakdown.inviscid_wings_sectional[0]
    
    # lift coefficient and sectional lift coefficient check
    lift_coefficient_true       = 0.6020886144805456 
    sectional_lift_coeff_true   = np.array([5.76806744e-01, 5.07127163e-01, 4.85174693e-01, 4.21180804e-01,
                                            8.20190073e-02, 5.80786371e-01, 5.26334659e-01, 4.88068047e-01,
                                            4.22435157e-01, 8.22818867e-02, 7.57930897e-03, 6.08409827e-03,
                                            4.28877462e-03, 3.85249022e-03, 2.76662518e-03, 4.51480220e-03,
                                            2.59270958e-03, 1.38993854e-03, 1.40831464e-03, 1.15658863e-03,
                                            2.05091472e-07, 9.99971822e-10, 1.12924489e-09, 2.64483046e-09,
                                            1.57775188e-09])

    diff_CL = np.abs(lift_coefficient  - lift_coefficient_true)
    print('CL difference')
    print(diff_CL)

    diff_Cl   = np.abs(sectional_lift_coeff - sectional_lift_coeff_true)

    print('Cl difference')
    print(diff_Cl)
    
    assert np.abs(lift_coefficient  - lift_coefficient_true) < 1e-6
    assert np.max(np.abs(sectional_lift_coeff - sectional_lift_coeff_true)) < 1e-6

    # plot results, vehicle, and vortex distribution
    plot_mission(results,configs.base)
    plot_3d_vehicle(configs.base, save_figure = True, plot_wing_control_points = False)
    plot_3d_vehicle_vlm_panelization(configs.base, save_figure=False, plot_wing_control_points=True)
              
    return
 

def plot_mission(results,vehicle): 
    
    # Plot lift distribution
    plot_lift_distribution(results,vehicle)

    return


# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def X57_setup(wake_fidelity, identical_props):

    # vehicle data
    vehicle  = vehicle_setup()
    # update wake method and rotation direction of rotors:
    props = vehicle.networks.battery_propeller.propellers
    for p in props:
        p.rotation = -1
        if wake_fidelity==1:
            p.Wake = Rotor_Wake_Fidelity_One()   
            p.Wake.wake_settings.number_rotor_rotations = 1  # reduced for regression speed
            p.Wake.wake_settings.number_steps_per_rotation = 24  # reduced for regression speed
            

    # test for non-identical propellers
    if not identical_props:
        vehicle.networks.battery_propeller.identical_propellers = False
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = X57_mission_setup(configs_analyses,vehicle)
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

def X57_mission_setup(analyses,vehicle):
    net_tag = list(vehicle.networks.keys())[0]
    
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
    base_segment.state.numerics.tolerance_solution           = 1e-10

    # ------------------------------------------------------------------
    #   Cruise Segment: constant Speed, constant altitude
    # ------------------------------------------------------------------
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    segment.analyses.extend(analyses.base)
    segment.battery_energy            = vehicle.networks[net_tag].battery.max_energy* 0.7
    segment.altitude                  = 8012 * Units.feet
    segment.air_speed                 = 115. * Units['mph']
    segment.distance                  = 20.  * Units.nautical_mile
    segment.state.unknowns.throttle   = 0.85 * ones_row(1)

    # post-process aerodynamic derivatives in cruise
    segment.process.finalize.post_process.aero_derivatives = SUAVE.Methods.Flight_Dynamics.Static_Stability.compute_aero_derivatives
    
    segment = vehicle.networks[net_tag].add_unknowns_and_residuals_to_segment(segment)

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


def Stopped_Rotor_vehicle(wake_fidelity, identical_props):

    # vehicle data
    vehicle  = V2()
    # update wake method and rotation direction of rotors:
    props = vehicle.networks.lift_cruise.propellers
    lift_rots = vehicle.networks.lift_cruise.lift_rotors
    for p in props:
        p.rotation = -1
        if wake_fidelity==1:
            p.Wake = Rotor_Wake_Fidelity_One()    
            p.Wake.wake_settings.number_rotor_rotations = 1  # reduced for regression speed
    for r in lift_rots:
        r.rotation = -1
        if wake_fidelity==1:
            r.Wake = Rotor_Wake_Fidelity_One()  
            r.Wake.wake_settings.number_rotor_rotations = 1  # reduced for regression speed      

    # test for non-identical propellers
    if not identical_props:
        vehicle.networks.lift_cruise.identical_propellers = False
        vehicle.networks.lift_cruise.identical_lift_rotors = False

    return vehicle


if __name__ == '__main__':
    main()
    plt.show()

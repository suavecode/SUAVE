# test_Multicopter.py
#
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke

""" setup file for a mission with an Electic Multicopter
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Aerodynamics.Vehicle import *  
from SUAVE.Visualization.Performance.Mission import *  
from SUAVE.Visualization.Performance.Energy.Common import *  
from SUAVE.Visualization.Performance.Energy.Battery import *   
from SUAVE.Visualization.Performance.Noise import *  
import numpy as np
import sys

sys.path.append('../Vehicles')
# the analysis functions

from Electric_Multicopter  import vehicle_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Electric Multicopter
    # ------------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    analyses.finalize()

    # Print weight properties of vehicle
    print(configs.base.weight_breakdown)
    print(configs.base.mass_properties.center_of_gravity)

    mission      = analyses.missions.base
    results      = mission.evaluate()

    # plot results
    plot_mission(results)

    # save, load and plot old results
    #save_multicopter_results(results)
    old_results = load_multicopter_results()
    plot_mission(old_results)

    # RPM of rotor check during hover
    RPM        = results.segments.climb.conditions.propulsion.propeller_rpm[0][0]
    RPM_true   = 1573.7516164319331

    print(RPM)
    diff_RPM = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3

    # Battery Energy Check During Transition
    battery_energy_transition         = results.segments.hover.conditions.propulsion.battery_energy[:,0]
    battery_energy_transition_true    = np.array([2.01217616e+08, 1.92155752e+08, 1.83082139e+08])

    print(battery_energy_transition)
    diff_battery_energy_transition    = np.abs(battery_energy_transition  - battery_energy_transition_true)
    print('battery energy of transition')
    print(diff_battery_energy_transition)
    assert all(np.abs((battery_energy_transition - battery_energy_transition_true)/battery_energy_transition_true) < 1e-3)


    return


# ----------------------------------------------------------------------
#   Setup
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


# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    return configs

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
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

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

    return analyses


def mission_setup(analyses,vehicle):


    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission     = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport            = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.airport    = airport

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.state.numerics.number_control_points        = 3

    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment                                               = Segments.Hover.Climb(base_segment)
    segment.tag                                           = "Climb"
    segment.analyses.extend( analyses.base)
    segment.altitude_start                                = 0.0  * Units.ft
    segment.altitude_end                                  = 40.  * Units.ft
    segment.climb_rate                                    = 300. * Units['ft/min']
    segment.battery_energy                                = vehicle.networks.battery_rotor.battery.max_energy
    segment.state.unknowns.throttle                       = 0.9 * ones_row(1)
    segment.process.iterate.conditions.stability          = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability       = SUAVE.Methods.skip
    segment = vehicle.networks.battery_rotor.add_unknowns_and_residuals_to_segment(segment,initial_power_coefficient=0.02)

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Hover Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment                                                 = Segments.Hover.Hover(base_segment)
    segment.tag                                             = "Hover"
    segment.analyses.extend( analyses.base)
    segment.altitude                                        = 40.  * Units.ft
    segment.time                                            = 2*60
    segment.process.iterate.conditions.stability            = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability         = SUAVE.Methods.skip
    segment = vehicle.networks.battery_rotor.add_unknowns_and_residuals_to_segment(segment)


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


# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results):

    # Plot Flight Conditions
    plot_flight_conditions(results)

    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results)

    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results)

    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results)

    # Plot Propeller Conditions
    plot_rotor_conditions(results)

    # Plot Electric Motor and Propeller Efficiencies
    plot_electric_motor_and_rotor_efficiencies(results)

    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results)

    return 


def load_multicopter_results():
    return SUAVE.Input_Output.SUAVE.load('results_multicopter.res')

def save_multicopter_results(results):
    SUAVE.Input_Output.SUAVE.archive(results,'results_multicopter.res')
    return

if __name__ == '__main__':
    main()

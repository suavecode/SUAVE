# test_Multicopter.py
#
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke

""" setup file for a mission with an Electic Multicopter
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import MARC
from MARC.Core import Units
from MARC.Visualization.Performance.Aerodynamics.Vehicle import *  
from MARC.Visualization.Performance.Mission import *  
from MARC.Visualization.Performance.Energy.Common import *  
from MARC.Visualization.Performance.Energy.Battery import *   
from MARC.Visualization.Performance.Noise import *  
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

    # RPM of rotor check during hover
    RPM        = results.segments.climb.conditions.propulsion.propulsor_group_0.rotor.rpm[0][0]
    RPM_true   = 1573.750927471253

    print(RPM)
    diff_RPM = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3

    # Battery Energy Check During Transition
    battery_energy_transition         = results.segments.hover.conditions.propulsion.battery.pack.energy[:,0]
    battery_energy_transition_true    = np.array([2.01220172e+08, 1.92176107e+08, 1.83120690e+08])

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

    analyses = MARC.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses

# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):
    analyses = MARC.Analyses.Analysis.Container()

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

    configs = MARC.Components.Configs.Config.Container()

    base_config = MARC.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    return configs

def base_analysis(vehicle):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------
    analyses = MARC.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = MARC.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = MARC.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Energy
    energy= MARC.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = MARC.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = MARC.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses


def mission_setup(analyses,vehicle):


    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission     = MARC.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport            = MARC.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = MARC.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.airport    = airport

    # unpack Segments module
    Segments = MARC.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row
    base_segment.process.iterate.initials.initialize_battery = MARC.Methods.Missions.Segments.Common.Energy.initialize_battery
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
    segment.battery_energy                                = vehicle.networks.battery_electric_rotor.battery.pack.max_energy 
    segment.process.iterate.conditions.stability          = MARC.Methods.skip
    segment.process.finalize.post_process.stability       = MARC.Methods.skip
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment,
                                                                                            initial_throttles = [0.90],
                                                                                            initial_rotor_power_coefficients=[0.02])

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
    segment.process.iterate.conditions.stability            = MARC.Methods.skip
    segment.process.finalize.post_process.stability         = MARC.Methods.skip
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment)


    # add to misison
    mission.append_segment(segment)

    return mission


def missions_setup(base_mission):

    # the mission container
    missions = MARC.Analyses.Mission.Mission.Container()

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
    plot_flight_conditions(results,show_figure=False)

    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results,show_figure=False)

    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results,show_figure=False)

    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results,show_figure=False)

    # Plot Propeller Conditions
    plot_rotor_conditions(results,show_figure=False)

    # Plot Electric Motor and Propeller Efficiencies
    plot_electric_motor_and_rotor_efficiencies(results,show_figure=False)

    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results,show_figure=False)

    return 

if __name__ == '__main__':
    main()

# test_Stopped_Rotor.py
#
# Created: Feb 2020, M. Clarke
#          Sep 2020, M. Clarke
#          Jul 2021, R. Erhard

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import MARC
from MARC.Core import Units , Data
from MARC.Visualization.Performance.Aerodynamics.Vehicle                 import *  
from MARC.Visualization.Performance.Mission                              import *  
from MARC.Visualization.Performance.Energy.Common                        import *  
from MARC.Visualization.Performance.Energy.Battery                       import *   
from MARC.Visualization.Performance.Noise                                import *  
from MARC.Visualization.Geometry.Three_Dimensional.plot_3d_vehicle       import plot_3d_vehicle 
from MARC.Visualization.Geometry                                         import *
from MARC.Methods.Performance.estimate_stall_speed                       import estimate_stall_speed
import sys
import numpy as np

sys.path.append('../Vehicles')
# the analysis functions

from Stopped_Rotor import vehicle_setup, configs_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():

    # ------------------------------------------------------------------------------------------------------------------
    # Stopped-Rotor
    # ------------------------------------------------------------------------------------------------------------------
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    configs.finalize()
    analyses.finalize()

    # Print weight properties of vehicle
    weights = configs.base.weight_breakdown
    print(weights)
    print(configs.base.mass_properties.center_of_gravity) 
    
    # check weights
    empty_thruth       = 1610.7272217061286
    structural_thruth  = 819.2186670281882
    total_thruth       = 1810.7272217061286
    rotors_thruth      = 23.038001631037044
    motors_thruth      = 38.0

    weights_error = Data()
    weights_error.empty       = abs(empty_thruth - weights.empty)/empty_thruth
    weights_error.structural  = abs(structural_thruth - weights.structural)/structural_thruth
    weights_error.total       = abs(total_thruth - weights.total)/total_thruth
    weights_error.lift_rotors = abs(rotors_thruth - weights.rotors)/rotors_thruth 
    weights_error.propellers  = abs(motors_thruth - weights.motors)/motors_thruth 

    for k, v in weights_error.items():
        assert (np.abs(v) < 1E-6)

    # evaluate mission
    mission   = analyses.missions.base
    results   = mission.evaluate()

    # plot results
    plot_mission(results) 

    # RPM of rotor check during hover
    RPM        = results.segments.climb_1.conditions.propulsion.propulsor_group_1.rotor.rpm[0][0]
    RPM_true   = 2403.00421420597
    print(RPM)
    diff_RPM   = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3

    # Battery Energy Check During Transition
    battery_energy_hover_to_transition      = results.segments.climb_2.conditions.propulsion.battery.pack.energy[:,0]
    battery_energy_hover_to_transition_true = np.array([3.19800413e+08, 3.18203003e+08, 3.16604274e+08])
    
    print(battery_energy_hover_to_transition)
    diff_battery_energy_hover_to_transition    = np.abs(battery_energy_hover_to_transition  - battery_energy_hover_to_transition_true)
    print('battery_energy_hover_to_transition difference')
    print(diff_battery_energy_hover_to_transition)
    assert all(np.abs((battery_energy_hover_to_transition - battery_energy_hover_to_transition_true)/battery_energy_hover_to_transition) < 1e-3)

    # lift Coefficient Check During Cruise
    lift_coefficient        = results.segments.departure_terminal_procedures.conditions.aerodynamics.lift_coefficient[0][0]
    lift_coefficient_true   = 0.8281462047156696

    print(lift_coefficient)
    diff_CL                 = np.abs(lift_coefficient  - lift_coefficient_true)
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3

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
    mission           = mission_setup(configs_analyses,vehicle)
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
    #  Aerodynamics Analysis
    aerodynamics = MARC.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= MARC.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Noise Analysis
    noise = MARC.Analyses.Noise.Fidelity_One() 
    noise.settings.level_ground_microphone_x_resolution = 2
    noise.settings.level_ground_microphone_y_resolution = 2       
    noise.geometry = vehicle
    analyses.append(noise)

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
    mission            = MARC.Analyses.Mission.Sequential_Segments()
    mission.tag        = 'the_mission'

    # airport
    airport            = MARC.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = MARC.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport    = airport

    # unpack Segments module
    Segments                                                 = MARC.Analyses.Mission.Segments

    # base segment
    base_segment                                             = Segments.Segment()
    base_segment.state.numerics.number_control_points        = 3 
    base_segment.process.initialize.initialize_battery       = MARC.Methods.Missions.Segments.Common.Energy.initialize_battery 
    ones_row                                                 = base_segment.state.ones_row
    
    # VSTALL Calculation  
    vehicle_mass   = vehicle.mass_properties.max_takeoff
    reference_area = vehicle.reference_area
    altitude       = 0.0 
    CL_max         = 1.2  
    Vstall         = estimate_stall_speed(vehicle_mass,reference_area,altitude,CL_max)      

    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.vertical_flight )
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 40.  * Units.ft
    segment.climb_rate                                       = 500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.networks.battery_electric_rotor.battery.pack.max_energy
    segment.process.iterate.unknowns.mission                 = MARC.Methods.skip
    segment.process.iterate.conditions.stability             = MARC.Methods.skip
    segment.process.finalize.post_process.stability          = MARC.Methods.skip   
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment,\
                                                                                    initial_rotor_power_coefficients = [0.02,0.01],
                                                                                    initial_throttles =  [0.7,0.9] )
    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------
    segment                                            = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag                                        = "transition_1"
    segment.analyses.extend( analyses.transition_flight )  
    segment.altitude                                   = 40.  * Units.ft
    segment.air_speed_start                            = 500. * Units['ft/min']
    segment.air_speed_end                              = 0.8 * Vstall
    segment.acceleration                               = 0.5 * Units['m/s/s']
    segment.pitch_initial                              = 0.0 * Units.degrees
    segment.pitch_final                                = 5. * Units.degrees  
    segment.process.iterate.unknowns.mission           = MARC.Methods.skip
    segment.process.iterate.conditions.stability       = MARC.Methods.skip
    segment.process.finalize.post_process.stability    = MARC.Methods.skip 
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment,
                                                         initial_rotor_power_coefficients = [ 0.2, 0.01],
                                                         initial_throttles = [0.95,0.9] )
    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------
    segment                                             = Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb(base_segment)
    segment.tag                                         = "transition_2"
    segment.analyses.extend( analyses.transition_flight )
    segment.altitude_start                          = 40.0 * Units.ft
    segment.altitude_end                            = 50.0 * Units.ft
    segment.climb_angle                             = 1 * Units.degrees
    segment.acceleration                            = 0.5 * Units['m/s/s']
    segment.pitch_initial                           = 5. * Units.degrees
    segment.pitch_final                             = 7. * Units.degrees 
    segment.process.iterate.unknowns.mission        = MARC.Methods.skip
    segment.process.iterate.conditions.stability    = MARC.Methods.skip
    segment.process.finalize.post_process.stability = MARC.Methods.skip
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment,
                                                         initial_rotor_power_coefficients = [ 0.2, 0.01],
                                                         initial_throttles = [0.95,0.9] )

    # add to misison
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment                                            = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag                                        = "climb_2"
    segment.analyses.extend( analyses.forward_flight)
    segment.air_speed                                  = 1.1*Vstall
    segment.altitude_start                             = 50.0 * Units.ft
    segment.altitude_end                               = 300. * Units.ft
    segment.climb_rate                                 = 500. * Units['ft/min']  
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment,
                                                         initial_throttles =[0.8,0.9] )

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Cruise Segment: Constant Speed, Constant Altitude
    # ------------------------------------------------------------------
    segment                                            = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag                                        = "departure_terminal_procedures"
    segment.analyses.extend( analyses.forward_flight )
    segment.altitude                                   = 300.0 * Units.ft
    segment.time                                       = 60.   * Units.second
    segment.air_speed                                  = 1.2*Vstall 
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment,\
                                                                                          initial_rotor_power_coefficients = [0.16,0.7],
                                                                                          initial_throttles = [0.8,0.9] )
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
 
    # Plot Electric Motor and Propeller Efficiencies  of Lift Cruise Network
    plot_electric_motor_and_rotor_efficiencies(results,show_figure=False)

    return
 
if __name__ == '__main__':
    main() 

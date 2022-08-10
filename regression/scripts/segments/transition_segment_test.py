# transition_segment_test.py
# 
# Created:  Mar 2022, R. Erhard
# Modified: 

""" setup file for transition segment test regression with a tiltrotor"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

from SUAVE.Methods.Performance.estimate_stall_speed import estimate_stall_speed
from SUAVE.Input_Output.VTK.save_vehicle_vtk import save_vehicle_vtks
from SUAVE.Plots.Performance import *
from SUAVE.Core import Data

import scipy as sp
import numpy as np
import pylab as plt
import sys
import os

sys.path.append('../Vehicles')
# the analysis functions

from Tiltrotor import vehicle_setup, configs_setup
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(): 
    # -----------------------------------------
    # Setup the vehicle configs and analyses
    # -----------------------------------------
    configs, analyses = full_setup() 
    configs.finalize()
    analyses.finalize()  
    
    # Evaluate the mission
    mission = analyses.missions.base
    results = mission.evaluate()
    
    # Plot results
    plot_mission(results,configs)
    

    # Generate vtks for animation of the tiltrotor through the transition segment
    #save_transition_animation_paraview(results,configs,save_path=None)       

    # Check throttles
    departure_throttle     = results.segments.departure.conditions.propulsion.throttle[:,0]
    transition_1_throttle  = results.segments.transition_1.conditions.propulsion.throttle[:,0]
    
    # Check network y-axis rotation during transition
    transition_y_axis_rotations = results.segments.transition_1.conditions.propulsion.propeller_y_axis_rotation[:,0]


    # Truth values
    departure_throttle_truth          = np.array([0.94582846, 0.94614831, 0.94681159, 0.94715503])
    transition_1_throttle_truth       = np.array([0.95751959, 0.9463189 , 0.8110942 , 0.6151161 ])
    transition_y_axis_rotations_truth = np.array([1.35008476, 1.32586108, 1.16978417, 0.82700895])

    # Store errors 
    error = Data()
    error.departure_throttle          = np.max(np.abs(departure_throttle - departure_throttle_truth))  
    error.transition_1_throttle       = np.max(np.abs(transition_1_throttle - transition_1_throttle_truth))   
    error.transition_y_axis_rotations = np.max(np.abs(transition_y_axis_rotations - transition_y_axis_rotations_truth))   
    
    #plt.show()  
    
    print('Errors:')
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)  

     
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
    for tag,config in list(configs.items()):
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
    weights = SUAVE.Analyses.Weights.Weights_eVTOL()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    analyses.append(aerodynamics)

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
    vehicle_mass       = vehicle.mass_properties.takeoff
    reference_area     = vehicle.reference_area
    V_stall            = estimate_stall_speed(vehicle_mass,reference_area,altitude=0*Units.feet,maximum_lift_coefficient=1.2)    
    max_vertical_rate  = 3.6 * Units['m/s'] 
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'tiltrotor_mission'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment                                       = Segments.Segment() 
    base_segment.state.numerics.number_control_points  = 4
    ones_row                                           = base_segment.state.ones_row 
    base_segment.process.initialize.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery

    # -------------------------------------------------------------------------
    #   Segment 0: Takeoff Vertically
    # -------------------------------------------------------------------------
    segment                                            = Segments.Hover.Climb(base_segment)
    segment.tag                                        = "Departure" 
    segment.analyses.extend( analyses.hover_climb)
    segment.altitude_start                             = 0.0  * Units.ft
    segment.altitude_end                               = 40.  * Units.ft
    segment.climb_rate                                 = 300. * Units['ft/min']
    segment.battery_energy                             = vehicle.networks.battery_propeller.battery.max_energy   
    segment.state.unknowns.throttle                    = 1.0 * ones_row(1) 
    segment.process.iterate.conditions.stability       = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability    = SUAVE.Methods.skip
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment,\
                                                                                         initial_power_coefficient = 0.06)
    # add to mission
    mission.append_segment(segment)

    # --------------------------------------------------------------------------
    #   Segment 1: First Transition Segment: Linear Speed, Constant Climb Rate
    # --------------------------------------------------------------------------
    # Use original transition segment, converge on rotor y-axis rotation and throttle
    segment                                             = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag                                         = "Transition_1"
    segment.analyses.extend( analyses.transition_1 )
    segment.altitude                                    = 40.0 * Units.ft
    segment.acceleration                                = 2.2  * Units['m/s/s']
    segment.air_speed_start                             = 1.1  * Units.mph              # starts from hover
    segment.air_speed_end                               = 1.   * V_stall         # increases linearly in time to stall speed
    segment.pitch_initial                               = 0.0  * Units.degrees  
    segment.pitch_final                                 = 3.6  * Units.degrees   
    segment.state.unknowns.throttle                     = 0.95  * ones_row(1)
    segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability     = SUAVE.Methods.skip
    segment = vehicle.networks.battery_propeller.add_tiltrotor_transition_unknowns_and_residuals_to_segment(segment, 
                                                                                                            initial_power_coefficient = 0.1)
    # add to misison
    mission.append_segment(segment)
    
    # --------------------------------------------------------------------------
    #   Segment 2a: Transition Segment: Linear Speed, Linear Climb
    # --------------------------------------------------------------------------
    # Use original transition segment, converge on rotor y-axis rotation and throttle
    segment                                             = Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb(base_segment)
    segment.tag                                         = "Transition_2a"
    segment.analyses.extend( analyses.transition_1 )
    segment.altitude_start                              = 40.0 * Units.ft
    segment.altitude_end                                = 100.0 * Units.ft
    segment.acceleration                                = 1.4  * Units['m/s/s']
    segment.climb_angle                                 = 5. * Units.deg
    segment.pitch_initial                               = 3.6  * Units.degrees  
    segment.pitch_final                                 = 4.0  * Units.degrees   
    segment.state.unknowns.throttle                     = 0.9  * ones_row(1)
    segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability     = SUAVE.Methods.skip
    segment = vehicle.networks.battery_propeller.add_tiltrotor_transition_unknowns_and_residuals_to_segment(segment, 
                                                                                                            initial_power_coefficient = 0.1)
    # add to misison
    mission.append_segment(segment)
    
    # --------------------------------------------------------------------------
    #   Segment 2b: Transition Segment: Linear Speed, Linear Climb
    # --------------------------------------------------------------------------
    # Use original transition segment, converge on rotor y-axis rotation and throttle
    segment                                             = Segments.Transition.Constant_Acceleration_Constant_Angle_Linear_Climb(base_segment)
    segment.tag                                         = "Transition_2b"
    segment.analyses.extend( analyses.transition_1 )
    segment.altitude_start                              = 100.0 * Units.ft
    segment.altitude_end                                = 90.0 * Units.ft 
    segment.acceleration                                = -0.1  * Units['m/s/s']
    segment.climb_angle                                 = 1. * Units.deg
    segment.pitch_initial                               = 4.0  * Units.degrees  
    segment.pitch_final                                 = 3.  * Units.degrees   
    segment.state.unknowns.throttle                     = 0.8  * ones_row(1)
    segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability     = SUAVE.Methods.skip
    segment = vehicle.networks.battery_propeller.add_tiltrotor_transition_unknowns_and_residuals_to_segment(segment, 
                                                                                                            initial_power_coefficient = 0.05)
    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   Mission definition complete    
    # ------------------------------------------------------------------ 

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
def plot_mission(results,configs,line_style = 'bo-'):  
    
    # Plot Flight Conditions 
    plot_flight_conditions(results, line_style) 
    
    # Plot Aerodynamic Coefficients
    plot_aerodynamic_coefficients(results, line_style)  
    
    # Plot Aircraft Flight Speed
    plot_aircraft_velocities(results, line_style)
    
    # Plot Aircraft Electronics
    plot_battery_pack_conditions(results, line_style)
    
    # Plot Propeller Conditions 
    plot_propeller_conditions(results, line_style) 
    
    # Plot Electric Motor and Propeller Efficiencies 
    plot_eMotor_Prop_efficiencies(results, line_style)
    
    # Plot tiltrotor conditions
    plot_tiltrotor_conditions(results,configs,line_style)

    # Plot propeller Disc and Power Loading
    plot_disc_power_loading(results, line_style)  

    return


def save_transition_animation_paraview(results,configs,save_path=None):
   
    # create store location
    if save_path == None:
        base_path = os.path.dirname(os.path.abspath(__file__)) 
        dirname   = base_path + "/Tiltrotor_VTKs/"
    else:
        dirname = save_path + "/Tiltrotor_VTKs/"
        
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("Directory " + dirname + " created.")        

    s_i = 0
    for s, seg in enumerate(results.segments):
        config = configs[list(configs.keys())[s]]
        y_rot = seg.conditions.propulsion.propeller_y_axis_rotation[:,0]
        time = seg.conditions.frames.inertial.time[:,0] 
        
        if seg.tag == "Transition_1":
            # expand for each second of transition
            t_duration = time[-1] - time[0]
            y_fun = sp.interpolate.interp1d(time,y_rot)
            new_times = np.linspace(time[0],time[-1], int(t_duration))
            new_y_rots = y_fun(new_times)
            
            for i in range(len(new_times)):
                config.networks.battery_propeller.y_axis_rotation = new_y_rots[i]
                # store vehicle for this control point
                save_vehicle_vtks(config,time_step=s_i, save_loc=dirname)
                s_i += 1       
                
    return


if __name__ == '__main__': 
    main()    
    
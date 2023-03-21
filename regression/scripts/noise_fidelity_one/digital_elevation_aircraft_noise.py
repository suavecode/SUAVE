# digital_elevation_aircraft_noise.py
#
# Created: Apr 2021, M. Clarke  
 
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import MARC
from MARC.Core import Units 
import numpy as np    
from MARC.Core import Data 
from MARC.Visualization.Performance.Aerodynamics.Vehicle import *  
from MARC.Visualization.Performance.Mission import *  
from MARC.Visualization.Performance.Energy.Common import *  
from MARC.Visualization.Performance.Energy.Battery import *   
from MARC.Visualization.Performance.Noise import *   
from MARC.Methods.Performance.estimate_stall_speed import estimate_stall_speed 
from MARC.Methods.Noise.Fidelity_One.Noise_Tools.generate_microphone_points import preprocess_topography_and_route_data
import matplotlib.pyplot as plt 

import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from X57_Maxwell_Mod2  import vehicle_setup as  X57_vehicle_setup
from X57_Maxwell_Mod2  import configs_setup as  X57_configs_setup   

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
    # ----------------------------------------------------------------------
    # MARC Frequency Domain Propeller Aircraft Noise Model 
    # ---------------------------------------------------------------------- 
    configs, analyses = X57_full_setup()  
    
    configs.finalize()
    analyses.finalize()   
    
    # mission analysis
    mission = analyses.missions.base
    X57_results = mission.evaluate()  
    
    # plot the results 
    plot_results(X57_results)  
    
    # SPL of rotor check during hover
    print('\n\n MARC Frequency Domain Propeller Aircraft Noise Model')

    X57_SPL        = np.max(X57_results.segments.departure_end_of_runway.conditions.noise.total_SPL_dBA) 
    X57_SPL_true   = 75.0624436062055
    
    print(X57_SPL) 
    X57_diff_SPL   = np.abs(X57_SPL - X57_SPL_true)
    print('SPL difference')
    print(X57_diff_SPL)
    assert np.abs((X57_SPL - X57_SPL_true)/X57_SPL_true) < 1e-3   # lower tolerance for highly machine tolerance sensitive computation 
    
    return     
 
# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------  
def X57_full_setup():

    # vehicle data
    vehicle  = X57_vehicle_setup()

    topography_data = preprocess_topography_and_route_data(topography_file                       = 'LA_Metropolitan_Area.txt',
                                                           departure_coordinates                 = [33.94067953101678, -118.40513722978149],
                                                           destination_coordinates               = [33.81713622114423, -117.92111163722772],
                                                           number_of_latitudinal_microphones     = 201,
                                                           number_of_longitudinal_microphones    = 101,
                                                           latitudinal_microphone_stencil_size   = 3,
                                                           longitudinal_microphone_stencil_size  = 3 )
    
    
    # change identical propeller flag for regression coverage even though propellers are identical 
    vehicle.networks.battery_electric_rotor.identical_rotors = False
    
    # Set up configs
    configs  = X57_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs,topography_data)

    # mission analyses
    mission  = X57_mission_setup(configs_analyses,vehicle,topography_data)
    missions_analyses = X57_missions_setup(mission)

    analyses = MARC.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses 

def base_analysis(vehicle,topography_data):

    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = MARC.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    # ------------------------------------------------------------------
    sizing = MARC.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    # ------------------------------------------------------------------
    weights = MARC.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    # ------------------------------------------------------------------
    aerodynamics = MARC.Analyses.Aerodynamics.Fidelity_Zero() 
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000 
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Noise Analysis 
    # ------------------------------------------------------------------   
    noise = MARC.Analyses.Noise.Fidelity_One()   
    noise.geometry = vehicle
    noise.settings.ground_microphone_x_resolution   = topography_data.ground_microphone_x_resolution           
    noise.settings.ground_microphone_y_resolution   = topography_data.ground_microphone_y_resolution          
    noise.settings.ground_microphone_x_stencil      = topography_data.ground_microphone_x_stencil             
    noise.settings.ground_microphone_y_stencil      = topography_data.ground_microphone_y_stencil             
    noise.settings.ground_microphone_min_y          = topography_data.ground_microphone_min_x                 
    noise.settings.ground_microphone_max_y          = topography_data.ground_microphone_max_x                 
    noise.settings.ground_microphone_min_x          = topography_data.ground_microphone_min_y                 
    noise.settings.ground_microphone_max_x          = topography_data.ground_microphone_max_y                 
    noise.settings.ground_microphone_locations      = topography_data.cartesian_microphone_locations            
    noise.settings.aircraft_departure_location      = topography_data.departure_location   
    noise.settings.aircraft_destimation_location    = topography_data.destination_location                              
    analyses.append(noise)                                                       
     
    # ------------------------------------------------------------------
    #  Energy
    # ------------------------------------------------------------------
    energy= MARC.Analyses.Energy.Energy()
    energy.network = vehicle.networks 
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    # ------------------------------------------------------------------
    planet = MARC.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    # ------------------------------------------------------------------
    atmosphere = MARC.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   

    # done!
    return analyses   
# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs,topography_data):

    analyses = MARC.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config,topography_data)
        analyses[tag] = analysis

    return analyses 


def simple_sizing(configs):

    base = configs.base
    base.pull_base()

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 1.75 * wing.areas.reference
        wing.areas.exposed  = 0.8  * wing.areas.wetted
        wing.areas.affected = 0.6  * wing.areas.wetted


    # diff the new data
    base.store_diff()

    # done!
    return
  

def X57_mission_setup(analyses,vehicle,topography_data):  
    

    # Determine Stall Speed 
    vehicle_mass   = vehicle.mass_properties.max_takeoff
    reference_area = vehicle.reference_area
    altitude       = 0.0 
    CL_max         = 1.2  
    Vstall         = estimate_stall_speed(vehicle_mass,reference_area,altitude,CL_max)   
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission = MARC.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
    airport = MARC.Attributes.Airports.Airport()
    airport.altitude   =  0. * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = MARC.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = MARC.Analyses.Mission.Segments 
    
    # base segment
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row 
    base_segment.process.initialize.initialize_battery       = MARC.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.state.numerics.number_control_points        = 4   
    
    # ------------------------------------------------------------------
    #   Departure End of Runway Segment Flight 1 : 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment) 
    segment.tag = 'Departure_End_of_Runway'    
    segment.analyses.extend(analyses.base)      
    segment.altitude_start                                   = 0.0 * Units.feet 
    segment.battery_energy                                   = vehicle.networks.battery_electric_rotor.battery.pack.max_energy    
    segment.altitude_end                                     = 50.0 * Units.feet
    segment.air_speed_start                                  = Vstall*1.1 
    segment.air_speed_end                                    = Vstall*1.2       
    segment.climb_rate                                       = 600 * Units['ft/min']
    segment.true_course_angle                                = topography_data.true_course_angle   
    segment = vehicle.networks.battery_electric_rotor.add_unknowns_and_residuals_to_segment(segment)  
    mission.append_segment(segment) 

    return mission

def X57_missions_setup(base_mission):

    # the mission container
    missions = MARC.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------
    missions.base = base_mission

    # done!
    return missions  

def plot_results(results):   
    
    # Plot noise level
    plot_ground_noise_levels(results)
    
    # Plot noise contour
    plot_flight_profile_noise_contours(results,show_figure=False)   
                        
    return  

if __name__ == '__main__': 
    main()    
    plt.show()

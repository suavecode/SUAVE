# aircraft_noise.py
#
# Created: Apr 2021, M. Clarke 

""" setup file for the X57-Maxwell Electric Aircraft to valdiate noise in a climb segment
"""
 
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np    
from SUAVE.Core import Data 
from SUAVE.Plots.Performance.Mission_Plots import *   
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from SUAVE.Methods.Noise.Certification import sideline_noise, flyover_noise, approach_noise
from SUAVE.Methods.Noise.Fidelity_One.Noise_Tools.generate_microphone_points import generate_building_microphone_points 
import matplotlib.pyplot as plt 

import sys

sys.path.append('../Vehicles')
# the analysis functions 
 
from X57_Maxwell_Mod2  import vehicle_setup as  X57_vehicle_setup
from X57_Maxwell_Mod2  import configs_setup as  X57_configs_setup  
from Boeing_737        import vehicle_setup as  B737_vehicle_setup 
from Boeing_737        import configs_setup as  B737_configs_setup 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
    # ----------------------------------------------------------------------
    # SUAVE Frequency Domain Propeller Aircraft Noise Model 
    # ---------------------------------------------------------------------- 
    configs, analyses = X57_full_setup() 
 
    configs.finalize()
    analyses.finalize()   
    
    # mission analysis
    mission = analyses.missions.base
    X57_results = mission.evaluate()  
    
    # plot the results
    X57_filename = "X57_Noise"
    plot_results(X57_results,X57_filename)  
    
    # SPL of rotor check during hover
    print('\n\n SUAVE Frequency Domain Propeller Aircraft Noise Model')
    X57_SPL        = X57_results.segments.ica.conditions.noise.total_SPL_dBA[3][0]
    X57_SPL_true   = 62.87188281671746

    print(X57_SPL) 
    X57_diff_SPL   = np.abs(X57_SPL - X57_SPL_true)
    print('SPL difference')
    print(X57_diff_SPL)
    assert np.abs((X57_SPL - X57_SPL_true)/X57_SPL_true) < 1e-6    
    
    # ----------------------------------------------------------------------
    # SAE Turbofan Aircraft Noise Model 
    # ---------------------------------------------------------------------- 
    configs, analyses = B737_full_setup() 

    configs.finalize()
    analyses.finalize()   
    
    # mission analysis
    mission       = analyses.missions.base
    B737_results  = mission.evaluate()  
    
    # certification calculations  
    sideline_SPL  = sideline_noise(analyses,configs) 
    flyover_SPL   = flyover_noise(analyses,configs)  
    approach_SPL  = approach_noise(analyses,configs) 
    
    # SPL of rotor check during hover
    print('\n\n SAE Turbofan Aircraft Noise Model')
    B737_SPL        = B737_results.segments.climb_1.conditions.noise.total_SPL_dBA[3][0]
    B737_SPL_true   = 27.760566836483797
    print(B737_SPL) 
    B737_diff_SPL   = np.abs(B737_SPL - B737_SPL_true)
    print('SPL difference')
    print(B737_diff_SPL)
    assert np.abs((B737_SPL - B737_SPL_true)/B737_SPL_true) < 1e-6    
    return     
 
# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------  
def X57_full_setup():

    # vehicle data
    vehicle  = X57_vehicle_setup()
    
    # Set up configs
    configs  = X57_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = X57_mission_setup(configs_analyses,vehicle)
    missions_analyses = X57_missions_setup(mission)

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses
 

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------  
def B737_full_setup():

    # vehicle data
    vehicle  = B737_vehicle_setup()
    vehicle.wings.main_wing.control_surfaces.flap.configuration_type = 'triple_slotted'  
    vehicle.wings.main_wing.high_lift = True
    vehicle.wings.main_wing = wing_planform(vehicle.wings.main_wing)
    
    # Set up configs
    configs  = B737_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = B737_mission_setup(configs_analyses)
    missions_analyses = B737_missions_setup(mission,configs_analyses )

    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses 

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
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000 
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Noise Analysis
    noise = SUAVE.Analyses.Noise.Fidelity_One()   
    noise.geometry = vehicle  
    urban_canyon_microphone_array,building_locations,building_dimensions,N_x,N_y,N_z = urban_canyon_microphone_setup() 
    noise.settings.urban_canyon_microphone_locations    = urban_canyon_microphone_array
    noise.settings.urban_canyon_building_locations      = building_locations
    noise.settings.urban_canyon_building_dimensions     = building_dimensions
    noise.settings.urban_canyon_microphone_x_resolution = N_x 
    noise.settings.urban_canyon_microphone_y_resolution = N_y
    noise.settings.urban_canyon_microphone_z_resolution = N_z      
    analyses.append(noise)

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
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------

def analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = base_analysis(config)
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
 
# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def urban_canyon_microphone_setup():  
    
    # define building locations 
    building_locations  = [[200,150,0],[400,-200,0]] # [[x,y,z]]     
     
    # define building dimensions  
    building_dimensions = [[100,200,75],[160,160,90]] # [[length,width,height]]     
    
    N_X = 4
    N_Y = 4
    N_Z = 16
    mic_locations  = generate_building_microphone_points(building_locations,building_dimensions,N_x = N_X ,N_y = N_Y ,N_z = N_Z ) 
     
    return mic_locations,building_locations ,building_dimensions,N_X ,N_Y ,N_Z 

def X57_mission_setup(analyses,vehicle):  
    
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
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row 
    base_segment.process.initialize.initialize_battery       = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.state.numerics.number_control_points        = 4  
    
    # ------------------------------------------------------------------
    #   Initial Climb Area Segment Flight 1  
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment) 
    segment.tag = 'ICA' 
    segment.analyses.extend( analyses.base )  
    segment.battery_energy                                   = vehicle.networks.battery_propeller.battery.max_energy  
    segment.state.unknowns.throttle                          = 0.85  * ones_row(1)  
    segment.altitude_start                                   = 50.0 * Units.feet
    segment.altitude_end                                     = 500.0 * Units.feet
    segment.air_speed_start                                  = 45  * Units['m/s']   
    segment.air_speed_end                                    = 50 * Units['m/s']   
    segment.climb_rate                                       = 600 * Units['ft/min']    
    
    segment = vehicle.networks.battery_propeller.add_unknowns_and_residuals_to_segment(segment)
    
    mission.append_segment(segment) 
              
    
    return mission

def B737_mission_setup(analyses): 
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'base_mission'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()


    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.takeoff )
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle = ones_row(1) * 5. * Units.deg      

    segment.altitude_start = 0.001   * Units.km
    segment.altitude_end   = 3.0   * Units.km
    segment.air_speed      = 125.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to misison
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    segment.analyses.extend( analyses.cutback )

    segment.altitude_end   = 8.0   * Units.km
    segment.air_speed      = 190.0 * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"

    segment.analyses.extend( analyses.base )

    segment.altitude_end = 10.668 * Units.km
    segment.air_speed    = 226.0  * Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses.base )

    segment.air_speed  = 230.412 * Units['m/s']
    segment.distance   = (3933.65 + 770 - 92.6) * Units.km

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"

    segment.analyses.extend( analyses.base )

    segment.altitude_end = 8.0   * Units.km
    segment.air_speed    = 220.0 * Units['m/s']
    segment.descent_rate = 4.5   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    segment.analyses.extend( analyses.base )

    segment.altitude_end = 6.0   * Units.km
    segment.air_speed    = 195.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_3"

    segment.analyses.extend( analyses.base )

    segment.altitude_end = 4.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Fourth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_4"

    segment.analyses.extend( analyses.base )

    segment.altitude_end = 2.0   * Units.km
    segment.air_speed    = 150.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']


    # add to mission
    mission.append_segment(segment)



    # ------------------------------------------------------------------
    #   Fifth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_5"

    segment.analyses.extend( analyses.landing)

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 3.0   * Units['m/s']


    # append to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete    
    # ------------------------------------------------------------------

    return mission 

def X57_missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------
    missions.base = base_mission

    # done!
    return missions  



def B737_missions_setup(base_mission,analyses):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------ 
    missions.base = base_mission


    # ------------------------------------------------------------------
    #   Mission for Constrained Fuel
    # ------------------------------------------------------------------    
    fuel_mission           = SUAVE.Analyses.Mission.Mission() 
    fuel_mission.tag       = 'fuel'
    fuel_mission.range     = 1277. * Units.nautical_mile
    fuel_mission.payload   = 19000.
    missions.append(fuel_mission)    


    # ------------------------------------------------------------------
    #   Mission for Constrained Short Field
    # ------------------------------------------------------------------    
    short_field            = SUAVE.Analyses.Mission.Mission(base_mission) 
    short_field.tag        = 'short_field'  
    
    airport                = SUAVE.Attributes.Airports.Airport()
    airport.altitude       =  0.0  * Units.ft
    airport.delta_isa      =  0.0
    airport.atmosphere     = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    airport.available_tofl = 1500.
    short_field.airport    = airport    
    missions.append(short_field)
    
    
    # ------------------------------------------------------------------
    #   Mission for Fixed Payload
    # ------------------------------------------------------------------    
    payload         = SUAVE.Analyses.Mission.Mission()  
    payload.tag     = 'payload'
    payload.range   = 2316. * Units.nautical_mile
    payload.payload = 19000.
    missions.append(payload)

    
    # ------------------------------------------------------------------
    #   Mission for Takeoff Noise
    # ------------------------------------------------------------------    
    takeoff                           = SUAVE.Analyses.Mission.Sequential_Segments()
    takeoff.tag                       = 'takeoff'   
                                      
    # airport                          
    airport                           = SUAVE.Attributes.Airports.Airport()
    airport.altitude                  =  0.0  * Units.ft
    airport.delta_isa                 =  0.0
    airport.atmosphere                = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    takeoff.airport                   = airport    

    # unpack Segments module
    Segments                          = SUAVE.Analyses.Mission.Segments 
    base_segment                      = Segments.Segment()  
    atmosphere                        = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet                            = SUAVE.Attributes.Planets.Earth() 
    
    # Climb Segment: Constant throttle, constant speed
    segment                           = Segments.Climb.Constant_Throttle_Constant_Speed(base_segment)
    segment.tag                       = "climb"    
    segment.analyses.extend(analyses.base ) 
    segment.atmosphere                = atmosphere    
    segment.planet                    = planet 
    segment.altitude_start            =  35. *  Units.fts
    segment.altitude_end              = 304.8 *  Units.meter
    segment.air_speed                 = 85.4 * Units['m/s']
    segment.throttle                  = 1.  
    ones_row                          = segment.state.ones_row  
    takeoff.append_segment(segment)

    # Cutback Segment: Constant speed, constant segment angle
    segment                           = Segments.Climb.Constant_Speed_Constant_Angle(base_segment)
    segment.tag                       = "cutback"   
    segment.atmosphere                = atmosphere    
    segment.planet                    = planet     
    segment.analyses.extend(analyses.base )
    segment.air_speed                 = 85.4 * Units['m/s']
    segment.climb_angle               = 2.86  * Units.degrees 
    takeoff.append_segment(segment)  
    
    # append mission 
    missions.append(takeoff)

    # ------------------------------------------------------------------
    #   Mission for Sideline Noise
    # ------------------------------------------------------------------     
    sideline_takeoff                  = SUAVE.Analyses.Mission.Sequential_Segments()
    sideline_takeoff.tag              = 'sideline_takeoff'   
    sideline_takeoff.airport          = airport  
    segment                           = Segments.Climb.Constant_Throttle_Constant_Speed(base_segment)
    segment.tag                       = "climb"    
    segment.analyses.extend(analyses.base)
    segment.atmosphere                = atmosphere    
    segment.planet                    = planet     
    segment.altitude_start            =  35. *  Units.fts
    segment.altitude_end              = 1600 *  Units.fts
    segment.air_speed                 = 85.4 * Units['m/s']
    segment.throttle                  = 1.  
    ones_row                          = segment.state.ones_row
    segment.state.unknowns.body_angle = ones_row(1) * 12. * Units.deg  
    segment.state.unknowns.wind_angle = ones_row(1) * 5. * Units.deg  
    sideline_takeoff.append_segment(segment)   
    
    missions.append(sideline_takeoff)
    
    # -------------------   -----------------------------------------------
    #   Mission for Landing Noise
    # ------------------------------------------------------------------    
    landing                           = SUAVE.Analyses.Mission.Sequential_Segments()
    landing.tag                       = 'landing'   
    landing.airport                   = airport      
    segment                           = Segments.Descent.Constant_Speed_Constant_Angle(base_segment)
    segment.tag                       = "descent"
    segment.analyses.extend(analyses.base ) 
    segment.atmosphere                = atmosphere    
    segment.planet                    = planet     
    segment.altitude_start            = 2.0   * Units.km
    segment.altitude_end              = 0.
    segment.air_speed                 = 67. * Units['m/s']
    segment.descent_angle             = 3.0   * Units.degrees  
    landing.append_segment(segment)
        
    missions.append(landing)
    
    return missions  

def plot_results(results,filename):   
    
    # Plot noise level
    plot_ground_noise_levels(results,save_filename = filename)
    
    # Plot noise contour
    plot_flight_profile_noise_contours(results,save_filename = filename + 'contour',show_figure=False)  # show figure set to false for regression.
                        
    return  

if __name__ == '__main__': 
    main()    
    plt.show()

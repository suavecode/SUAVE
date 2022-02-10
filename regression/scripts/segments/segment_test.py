# segment_test.py
# 
# Created:  Feb 2020, M. Clarke
#           Apr 2020, M. Clarke
# Modified: Dec 2020, S. Karpuk
#         : Jan 2022, S. Claridge

""" setup file for segment test regression with a Boeing 737 and a modified aircraft that uses additional cryogenic fuel """

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np
import pylab as plt


from SUAVE.Core import Data
from SUAVE.Methods.Center_of_Gravity.compute_component_centers_of_gravity import compute_component_centers_of_gravity

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup, configs_setup

from Additional_Fuel_Aircraft import vehicle_setup as cf_vehicle_setup
from Additional_Fuel_Aircraft import configs_setup as cf_configs_setup

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main(): 

    # Truth values for Boeing 737 and cryogenic fuel aircraft
    boeing_737_results = [["climb_throttle_1_true" , 1.076870435474248],
                            ["climb_throttle_2_true", 1.0811705275252188],
                            ["climb_throttle_3_true", 0.6708679936668738],
                            ["climb_throttle_4_true", 1.1265503689284677], 
                            ["climb_4_total_mass_true", 77954.17842536],
                            ["climb_4_vehicle_additional_fuel_rate_true", 0.0],
                            ["climb_throttle_5_true", 1.1770721802348285],
                            ["climb_throttle_6_true", 0.7772366352514147], 
                            ["climb_throttle_7_true", 0.9754619978463199], 
                            ["climb_throttle_8_true", 1.2263494140483848],
                            ["climb_throttle_9_true", 1.339647400470607], 
                            ["climb_throttle_10_true", 1.0],
                            ["cruise_CL_1_true", 0.6972564737884285],
                            ["cruise_CL_2_true", 0.6975787113223217],
                            ["cruise_CL_3_true", 0.7852502026240504],
                            ["descent_throttle_1_true", 0.09102457856960339],
                            ["descent_throttle_2_true", 0.24465051386674283], 
                            ["single_pt_CL_1_true", 0.251138262389197],
                            ["single_pt_CL_2_true", 0.25113968108269286],
                            ["loiter_CL_true", 0.5114262749651366],
                            ["descent_throttle_3_true", 0.17797156049307095],
                            ["descent_3_total_mass_true", 72782.00046267 ],
                            ["descent_3_vehicle_additional_fuel_rate_true", 0.0]]

    boeing_737_vehicle  = vehicle_setup()
    boening_737_configs = configs_setup(boeing_737_vehicle)

    aircraft_using_cryogenic_fuel  = [["climb_throttle_1_true" , 2.09836518],
                        ["climb_throttle_2_true", 1.81422613],
                        ["climb_throttle_3_true", 1.01898036],
                        ["climb_throttle_4_true", 1.48489388], 
                        ["climb_4_total_mass_true", 78399.62679503],
                        ["climb_4_vehicle_additional_fuel_rate", 0.01710773],
                        ["climb_throttle_5_true", 1.47404658],
                        ["climb_throttle_6_true", 0.9207219], 
                        ["climb_throttle_7_true",  1.07734465], 
                        ["climb_throttle_8_true", 1.29949068],
                        ["climb_throttle_9_true", 1.29171816], 
                        ["climb_throttle_10_true", 1.0],
                        ["cruise_CL_1_true",  0.70703183],
                        ["cruise_CL_2_true",  0.70770214],
                        ["cruise_CL_3_true",  0.91800812],
                        ["descent_throttle_1_true", 0.10790657],
                        ["descent_throttle_2_true", 0.46200369], 
                        ["single_pt_CL_1_true", 0.25615682 ],
                        ["single_pt_CL_2_true", 0.25598024],
                        ["loiter_CL_true", 0.52161734],
                        ["descent_throttle_3_true",  0.35119837],
                        ["descent_3_total_mass_true", 74129.2794989  ],
                        ["descent_3_vehicle_additional_fuel_rate", 0.01710773]]   

    vehicle_using_cryogenic_fuel         = cf_vehicle_setup()
    vehicle_using_cryogenic_fuel_configs = cf_configs_setup(vehicle_using_cryogenic_fuel)

    aircraft_results = [[boeing_737_vehicle, boening_737_configs, boeing_737_results], [vehicle_using_cryogenic_fuel, vehicle_using_cryogenic_fuel_configs, aircraft_using_cryogenic_fuel]]

    for aircraft_result in aircraft_results:

        vehicle         = aircraft_result[0]
        config          = aircraft_result[1]
        true_results    = aircraft_result[2]

        results = get_results(vehicle, config)

        # Extract sample values from computation 
        climb_throttle_1                        = results.segments.climb_1.conditions.propulsion.throttle[3][0]
        climb_throttle_2                        = results.segments.climb_2.conditions.propulsion.throttle[3][0]
        climb_throttle_3                        = results.segments.climb_3.conditions.propulsion.throttle[3][0]
        climb_throttle_4                        = results.segments.climb_4.conditions.propulsion.throttle[3][0]
        climb_total_mass_4                      = results.segments.climb_4.conditions.weights.total_mass[0][0]
        climb_vehicle_additional_fuel_rate_4    = results.segments.climb_4.conditions.weights.vehicle_additional_fuel_rate[0][0]
        climb_throttle_5                        = results.segments.climb_5.conditions.propulsion.throttle[3][0]
        climb_throttle_6                        = results.segments.climb_6.conditions.propulsion.throttle[3][0]
        climb_throttle_7                        = results.segments.climb_7.conditions.propulsion.throttle[3][0] 
        climb_throttle_8                        = results.segments.climb_8.conditions.propulsion.throttle[3][0] 
        climb_throttle_9                        = results.segments.climb_9.conditions.propulsion.throttle[3][0] 
        climb_throttle_10                       = results.segments.climb_10.conditions.propulsion.throttle[2][0] 
        cruise_CL_1                             = results.segments.cruise_1.conditions.aerodynamics.lift_coefficient[2][0]
        cruise_CL_2                             = results.segments.cruise_2.conditions.aerodynamics.lift_coefficient[2][0]
        cruise_CL_3                             = results.segments.cruise_3.conditions.aerodynamics.lift_coefficient[2][0] 
        descent_throttle_1                      = results.segments.descent_1.conditions.propulsion.throttle[3][0]
        descent_throttle_2                      = results.segments.descent_2.conditions.propulsion.throttle[3][0]
        single_pt_CL_1                          = results.segments.single_point_1.conditions.aerodynamics.lift_coefficient[0][0]
        single_pt_CL_2                          = results.segments.single_point_2.conditions.aerodynamics.lift_coefficient[0][0]     
        loiter_CL                               = results.segments.loiter.conditions.aerodynamics.lift_coefficient[2][0]
        descent_throttle_3                      = results.segments.descent_3.conditions.propulsion.throttle[3][0]
        descent_total_mass_3                    = results.segments.descent_3.conditions.weights.total_mass[0][0]
        descent_vehicle_additional_fuel_rate_3  = results.segments.descent_3.conditions.weights.vehicle_additional_fuel_rate[0][0]


        check_results = np.array([climb_throttle_1, climb_throttle_2, climb_throttle_3, climb_throttle_4, climb_total_mass_4, climb_vehicle_additional_fuel_rate_4, 
                                climb_throttle_5, climb_throttle_6, climb_throttle_7, climb_throttle_8, climb_throttle_9, climb_throttle_10, cruise_CL_1, cruise_CL_2,
                                cruise_CL_3, descent_throttle_1, descent_throttle_2, single_pt_CL_1, single_pt_CL_2, loiter_CL, descent_throttle_3,
                                descent_total_mass_3, descent_vehicle_additional_fuel_rate_3 ])

        #print values for resetting regression
        print(*check_results, sep='\n')

        #create array of truth values 
        compare_true_results = np.array([item[1] for item in true_results])

        #calculate error values 
        error_val            = (compare_true_results - check_results) / compare_true_results
        error_val            = np.where(np.isnan(error_val), 0, error_val) #incase there was any divison by 0 

        assert(np.amax(error_val) < 1e-6)

    plt.show()    


def get_results(vehicle_setup, configs_setup):

    # -----------------------------------------
    # Multi-Point Mission Setup 
    # -----------------------------------------
    configs, analyses = full_setup(vehicle_setup, configs_setup) 
    simple_sizing(configs, analyses) 
    configs.finalize()
    analyses.finalize()  
    mission = analyses.missions.base
    results = mission.evaluate()

    return results

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------

def full_setup(vehicle, configs):

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses)
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

    # adjust analyses for configs

    # takeoff_analysis
    analyses.takeoff.aerodynamics.settings.drag_coefficient_increment = 0.0000

    # landing analysis
    aerodynamics = analyses.landing.aerodynamics

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
    aerodynamics.geometry = vehicle
    aerodynamics.settings.number_spanwise_vortices   = 5
    aerodynamics.settings.number_chordwise_vortices  = 2       
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
    energy.network = vehicle.networks #what is called throughout the mission (at every time step))
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

def simple_sizing(configs, analyses):

    base = configs.base
    base.pull_base()

    # zero fuel weight
    base.mass_properties.max_zero_fuel = 0.9 * base.mass_properties.max_takeoff 

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted

    # fuselage seats
    base.fuselages['fuselage'].number_coach_seats = base.passengers
    
    # weight analysis
    #need to put here, otherwise it won't be updated
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()    
    
    
    #compute centers of gravity
    #need to put here, otherwise, results won't be stored
    compute_component_centers_of_gravity(base)
    base.center_of_gravity()
    
    # diff the new data
    base.store_diff()

    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = configs.landing

    # make sure base data is current
    landing.pull_base()

    # landing weight
    landing.mass_properties.landing = 0.85 * base.mass_properties.takeoff

    # diff the new data
    landing.store_diff()

    # done!
    return

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------

def mission_setup(analyses): 

    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

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
    
    ones_row     = base_segment.state.ones_row 
    base_segment.state.numerics.number_control_points = 4
    
    # ------------------------------------------------------------------
    #   Takeoff Roll
    # ------------------------------------------------------------------

    segment = Segments.Ground.Takeoff(base_segment)
    segment.tag = "Takeoff"

    segment.analyses.extend( analyses.takeoff )
    segment.velocity_start           = 100.* Units.knots
    segment.velocity_end             = 150 * Units.knots
    segment.friction_coefficient     = 0.04
    segment.altitude                 = 0.0

    # add to misison
    mission.append_segment(segment)
    

    # ------------------------------------------------------------------
    #    Climb 1 : Constant Speed Constant Rate  
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"

    segment.analyses.extend( analyses.takeoff )

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 0.05  * Units.km
    segment.air_speed      = 150   * Units.knots
    segment.climb_rate     = 10.0  * Units['m/s']

    # add to misison
    mission.append_segment(segment)

   
    # ------------------------------------------------------------------
    #   Climb 2 : Constant Dynamic Pressure Constant Angle 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Dynamic_Pressure_Constant_Angle(base_segment)
    segment.tag = "climb_2"
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start                   = 0.05  * Units.km
    segment.altitude_end                     = 2.    * Units.km
    segment.climb_angle                      = 5.   * Units.degrees 
    segment.dynamic_pressure                 = 3800 * Units.pascals   

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Climb 3 : Constant Dynamic Pressure Constant Rate 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Dynamic_Pressure_Constant_Rate(base_segment)
    segment.tag = "climb_3"
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start                   = 2.   * Units.km
    segment.altitude_end                     = 4.   * Units.km
    segment.climb_rate                       = 730. * Units['ft/min']    
    segment.dynamic_pressure                 = 12000 * Units.pascals    

    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Climb 4 : Constant Mach Constant Angle 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Mach_Constant_Angle(base_segment)
    segment.tag = "climb_4"
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start                   = 4.   * Units.km
    segment.altitude_end                     = 6.   * Units.km
    segment.mach                             = 0.5
    segment.climb_angle                      = 3.5 * Units.degrees  

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Climb 5 : 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Constant_Angle(base_segment)
    segment.tag = "climb_5"
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start                   = 6.    * Units.km
    segment.altitude_end                     = 7.    * Units.km
    segment.air_speed                        = 180   * Units.m / Units.s
    segment.climb_angle                      = 3.    * Units.degrees    

    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Climb 6 : 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Mach_Linear_Altitude(base_segment)
    segment.tag = "climb_6"
    segment.analyses.extend( analyses.base )  
    segment.altitude_end                     = 8.    * Units.km   
    segment.mach                             = 0.75  

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Climb 7 : Constant Speed Linear Altitude 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_Speed_Linear_Altitude(base_segment)
    segment.tag = "climb_7"
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start                   = 8.    * Units.km
    segment.altitude_end                     = 9.    * Units.km   
    segment.air_speed                        = 250.2 * Units.m / Units.s 

    # add to misison
    mission.append_segment(segment)
 
    # ------------------------------------------------------------------
    #   Climb 8 : Constant EAS Constant Rate 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_EAS_Constant_Rate(base_segment)
    segment.tag = "climb_8"
    segment.analyses.extend( analyses.base )  
    segment.altitude_end                     = 10.   * Units.km    
    segment.equivalent_air_speed             = 150. * Units.m / Units.s
    segment.climb_rate                       = 1.  
    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Climb 9 : Constant EAS Constant Rate 
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Constant_CAS_Constant_Rate(base_segment)
    segment.tag = "climb_9"
    segment.analyses.extend( analyses.base )  
    segment.altitude_end                     = 10.9   * Units.km    
    segment.calibrated_air_speed             = 150. * Units.m / Units.s
    segment.climb_rate                       = 1.  
    # add to misison
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Climb 10 : Optimized
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Optimized(base_segment)
    segment.tag = "climb_10"
    segment.analyses.extend( analyses.base )  
    segment.altitude_start         = 10.9   * Units.km   
    segment.altitude_end           = 11.0   * Units.km   
    segment.air_speed_start        = 160. * Units.m / Units.s
    segment.air_speed_end          = None
    segment.objective              = 'conditions.frames.inertial.time[-1,0]*1000'
    segment.minimize               = True
    segment.state.numerics.number_control_points = 3
    # add to misison
    mission.append_segment(segment)
        
    
    # ------------------------------------------------------------------
    #   Cruise Segment 1: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Dynamic_Pressure_Constant_Altitude(base_segment)
    segment.tag = "cruise_1" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 11. * Units.km    
    segment.dynamic_pressure          = 28000 * Units.pascals   
    segment.distance                  = 500 * Units.km 
    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Cruise Segment 2: Constant Throttle Constant Altltude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Throttle_Constant_Altitude(base_segment)
    segment.tag = "cruise_2" 
    segment.analyses.extend(analyses.base)   
    
    segment.air_speed_end             = 200 * Units.m / Units.s 
    segment.throttle                  = 0.6
    segment.distance                  = 500 * Units.km 
    segment.state.numerics.number_control_points = 16
    
    # add to misison
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------
    #   Cruise Segment 3 : Constant Pitch Rate Constant Altltude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Pitch_Rate_Constant_Altitude(base_segment)
    segment.tag = "cruise_3" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 10. * Units.km    
    segment.pitch_rate                = 0.0001  * Units['rad/s/s']
    segment.pitch_final               = 4.  * Units.degrees 
    segment.distance                  = 500 * Units.km 
    segment.state.unknowns.throttle = ones_row(1) * 0.9
    segment.state.unknowns.velocity = ones_row(1) * 200
    # add to misison
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------
    #   Descent Segment 1: Constant Speed Constant Angle 
    # ------------------------------------------------------------------ 
    segment = Segments.Descent.Constant_Speed_Constant_Angle(base_segment)
    segment.tag = "descent_1" 
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start            = 10.    * Units.km    
    segment.air_speed                 = 150   * Units.m / Units.s 
    segment.altitude_end              = 5  * Units.km 
    
    # add to misison
    mission.append_segment(segment) 

    # ------------------------------------------------------------------
    #   Descent Segment 2: Constant CAS Constant Angle 
    # ------------------------------------------------------------------ 
    segment = Segments.Descent.Constant_CAS_Constant_Rate(base_segment)
    segment.tag = "descent_2" 
    segment.analyses.extend( analyses.base ) 
    segment.altitude_end              = 2500. * Units.feet
    segment.descent_rate              = 2.  * Units.m / Units.s
    segment.calibrated_air_speed      = 100 * Units.m / Units.s
    
    # add to misison
    mission.append_segment(segment) 

    # ------------------------------------------------------------------
    #  Single Point Segment 1: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Single_Point.Set_Speed_Set_Altitude(base_segment)
    segment.tag = "single_point_1" 
    segment.analyses.extend(analyses.base) 
    segment.altitude    =  2500. * Units.feet
    segment.air_speed   =  200. * Units['m/s']  

    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #  Single Point Segment 1: constant Speed, constant altitude
    # ------------------------------------------------------------------ 
    segment = Segments.Single_Point.Set_Speed_Set_Throttle(base_segment)
    segment.tag = "single_point_2" 
    segment.analyses.extend(analyses.base) 
    segment.altitude    =  2500. * Units.feet
    segment.air_speed   =  200. * Units['m/s']   
    segment.throttle    =  0.5

    # add to misison
    mission.append_segment(segment)    
    
    # ------------------------------------------------------------------
    #   Loiter Segment: Constant Dynamic Pressure Constant Altitude Loiter
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Dynamic_Pressure_Constant_Altitude_Loiter(base_segment)
    segment.tag = "loiter" 
    segment.analyses.extend(analyses.base) 
    segment.altitude                  = 2500  * Units.feet
    segment.dynamic_pressure          = 12000 * Units.pascals 
    # add to misison
    mission.append_segment(segment)   
       
    # ------------------------------------------------------------------
    #   Descent Segment: Constant EAS Constant Rate
    # ------------------------------------------------------------------ 
    segment = Segments.Descent.Constant_EAS_Constant_Rate(base_segment)
    segment.tag = "descent_3" 
    segment.analyses.extend( analyses.base ) 
    segment.altitude_start            = 2500  * Units.feet
    segment.altitude_end              = 0  * Units.feet 
    segment.descent_rate              = 3.  * Units.m / Units.s
    segment.equivalent_air_speed      = 100 * Units.m / Units.s
    
    # add to misison
    mission.append_segment(segment)   

    # ------------------------------------------------------------------
    #   Landing Roll
    # ------------------------------------------------------------------ 

    segment = Segments.Ground.Landing(base_segment)
    segment.tag = "Landing"

    segment.analyses.extend( analyses.landing )
    segment.velocity_start           = 150 * Units.knots
    segment.velocity_end             = 100 * Units.knots
    segment.friction_coefficient     = 0.4
    segment.altitude                 = 0.0

    # add to misison
    mission.append_segment(segment)     


    # ------------------------------------------------------------------
    #   Non Converged Segment : Constant Throttle Constant Altltude
    # ------------------------------------------------------------------ 
    segment = Segments.Cruise.Constant_Throttle_Constant_Altitude(base_segment)
    segment.tag = "cruise_non_converged" 
    segment.analyses.extend(analyses.base)   
    
    segment.air_speed_end                        = 150 * Units.knots
    segment.throttle                             = 0
    segment.distance                             = 10 * Units.km 
    segment.state.numerics.number_control_points = 2
    segment.state.numerics.max_evaluations       = 10
    
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

    return  
if __name__ == '__main__': 
    main()    

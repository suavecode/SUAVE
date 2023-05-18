# B737_noise.py
#
# Created: Apr 2021, M. Clarke 

""" setup file for the X57-Maxwell Electric Aircraft to valdiate noise in a climb segment
"""
 
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import MARC
from MARC.Core import Units 
import numpy as np    
from MARC.Core import Data 
from MARC.Visualization import *     
from MARC.Methods.Performance.estimate_stall_speed import estimate_stall_speed
from MARC.Methods.Geometry.Two_Dimensional.Planform import wing_planform
from MARC.Methods.Noise.Certification import sideline_noise, flyover_noise, approach_noise 
import matplotlib.pyplot as plt 

import sys

sys.path.append('../Vehicles')
# the analysis functions 
    
from Boeing_737        import vehicle_setup as  B737_vehicle_setup 
from Boeing_737        import configs_setup as  B737_configs_setup 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():    
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
    B737_SPL_true   = 21.799191185936067
    print(B737_SPL) 
    B737_diff_SPL   = np.abs(B737_SPL - B737_SPL_true)
    print('SPL difference')
    print(B737_diff_SPL)
    assert np.abs((B737_SPL - B737_SPL_true)/B737_SPL_true) < 1e-6    
    return     
  

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

    analyses = MARC.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses

    return configs, analyses 

def base_analysis(vehicle):

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
    noise = MARC.Analyses.Noise.Fidelity_Zero()   
    noise.geometry = vehicle          
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

def analyses_setup(configs):

    analyses = MARC.Analyses.Analysis.Container()

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

def B737_mission_setup(analyses): 
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = MARC.Analyses.Mission.Sequential_Segments()
    mission.tag = 'base_mission'

    #airport
    airport = MARC.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = MARC.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = MARC.Analyses.Mission.Segments

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

def B737_missions_setup(base_mission,analyses):

    # the mission container
    missions = MARC.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------ 
    missions.base = base_mission


    # ------------------------------------------------------------------
    #   Mission for Constrained Fuel
    # ------------------------------------------------------------------    
    fuel_mission           = MARC.Analyses.Mission.Mission() 
    fuel_mission.tag       = 'fuel'
    fuel_mission.range     = 1277. * Units.nautical_mile
    fuel_mission.payload   = 19000.
    missions.append(fuel_mission)    


    # ------------------------------------------------------------------
    #   Mission for Constrained Short Field
    # ------------------------------------------------------------------    
    short_field            = MARC.Analyses.Mission.Mission(base_mission) 
    short_field.tag        = 'short_field'  
    
    airport                = MARC.Attributes.Airports.Airport()
    airport.altitude       =  0.0  * Units.ft
    airport.delta_isa      =  0.0
    airport.atmosphere     = MARC.Attributes.Atmospheres.Earth.US_Standard_1976()
    airport.available_tofl = 1500.
    short_field.airport    = airport    
    missions.append(short_field)
    
    
    # ------------------------------------------------------------------
    #   Mission for Fixed Payload
    # ------------------------------------------------------------------    
    payload         = MARC.Analyses.Mission.Mission()  
    payload.tag     = 'payload'
    payload.range   = 2316. * Units.nautical_mile
    payload.payload = 19000.
    missions.append(payload)

    
    # ------------------------------------------------------------------
    #   Mission for Takeoff Noise
    # ------------------------------------------------------------------    
    takeoff                           = MARC.Analyses.Mission.Sequential_Segments()
    takeoff.tag                       = 'takeoff'   
                                      
    # airport                          
    airport                           = MARC.Attributes.Airports.Airport()
    airport.altitude                  =  0.0  * Units.ft
    airport.delta_isa                 =  0.0
    airport.atmosphere                = MARC.Analyses.Atmospheric.US_Standard_1976()
    takeoff.airport                   = airport    

    # unpack Segments module
    Segments                          = MARC.Analyses.Mission.Segments 
    base_segment                      = Segments.Segment()  
    atmosphere                        = MARC.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet                            = MARC.Attributes.Planets.Earth() 
    
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
    sideline_takeoff                  = MARC.Analyses.Mission.Sequential_Segments()
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
    landing                           = MARC.Analyses.Mission.Sequential_Segments()
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

if __name__ == '__main__': 
    main()    
    plt.show()

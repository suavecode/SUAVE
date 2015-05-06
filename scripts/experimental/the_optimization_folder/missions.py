# example_test_script.py
# 
# Created:  May 2015, T. Lukaczyk
# Modified: 


# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units

import numpy as np

# ----------------------------------------------------------------------
#   Define the Mission
# ----------------------------------------------------------------------
    
def setup(analyses):
    
    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()
    
    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------
    
    missions.base = base(analyses)
    
    
    # ------------------------------------------------------------------
    #   Mission for Constrained Fuel
    # ------------------------------------------------------------------    
    fuel_mission = SUAVE.Analyses.Mission.Mission() #Fuel_Constrained()
    fuel_mission.tag = 'fuel'
    fuel_mission.mission = base_mission
    fuel_mission.range   = 1277. * Units.nautical_mile
    fuel_mission.payload   = 19000.
    missions.append(fuel_mission)    
    
    
    # ------------------------------------------------------------------
    #   Mission for Constrained Short Field
    # ------------------------------------------------------------------    
    short_field = SUAVE.Analyses.Mission.Mission() #Short_Field_Constrained()
    short_field.mission = base_mission
    short_field.tag = 'short_field'    
    
    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    airport.available_tofl = 1500.
    short_field.mission.airport = airport    
    missions.append(short_field)
   
   
    # ------------------------------------------------------------------
    #   Mission for Fixed Payload
    # ------------------------------------------------------------------    
    payload = SUAVE.Analyses.Mission.Mission() #Payload_Constrained()
    payload.mission = base_mission
    payload.tag = 'payload'
    payload.range   = 2316. * Units.nautical_mile
    payload.payload   = 19000.
    missions.append(payload)
    
    
    # done!
    return missions  

    
def base(analyses):
    
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
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    
    segment.analyses.extend( analyses.takeoff )
    
    segment.altitude_start = 0.0   * Units.km
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
    
    segment.analyses.extend( analyses.cruise )
    
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
    
    segment.analyses.extend( analyses.cruise )
    
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
    
    segment.analyses.extend( analyses.cruise )
    
    segment.air_speed  = 230.412 * Units['m/s']
    segment.distance   = 3933.65 * Units.km
        
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    
    
    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end = 5.0   * Units.km
    segment.air_speed    = 170.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    

    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    segment.analyses.extend( analyses.landing )
    
    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 145.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    from vehicles import setup as vehicles_setup
    from analyses import setup as analyses_setup
    missions_setup = setup
    
    vehicles = vehicles_setup()
    analyses = analyses_setup(vehicles)
    missions = missions_setup(analyses)
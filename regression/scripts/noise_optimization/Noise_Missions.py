# Noise_Missions.py
# 
# Created:  May 2015, T. Lukaczyk
# Modified: Jan 2016, Carlos/Tarik
#           Apr 2020, M. Clarke

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
    base_mission = base(analyses)  # mission for performance
    missions.base = base_mission 
    
    # Maximum range mission
    mission_max_range = max_range_setup(analyses)  # mission for MTOW
    missions.max_range = mission_max_range     

    # Short field mission
    mission_short_field = short_field_setup(analyses)  # mission for Short Field
    missions.short_field = mission_short_field    
    
    # Takeoff mission initialization
    #missions.takeoff_initialization = takeoff_mission_initialization(analyses)
    
    # Takeoff mission
    missions.takeoff = takeoff_mission_setup(analyses)
    
    # Sideline Takeoff mission
    missions.sideline_takeoff = sideline_mission_setup(analyses)    
       
    # Landing mission
    missions.landing = landing_mission_setup(analyses)        

    # ------------------------------------------------------------------
    #   Mission for Takeoff Field Lengths
    # ------------------------------------------------------------------    
    takeoff = SUAVE.Analyses.Mission.Mission(base_mission) #Short_Field_Constrained()
    takeoff.tag = 'takeoff_field'    

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   = 0.0  
    airport.delta_isa  = 0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()

    takeoff.airport = airport    
    missions.append(takeoff)
    
    return missions  

    
def base(analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'base'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   = 0.0  
    airport.delta_isa  = 0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment() 
    base_segment.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
    
    atmosphere=SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
     # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.base )

    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.048 * Units.km
    segment.air_speed      = 138.0 * Units['m/s']
    segment.climb_rate     = 2900. * Units['ft/min']

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 168.0 * Units['m/s']
    segment.climb_rate   = 2500. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 25000. * Units.ft
    segment.air_speed    = 200.0  * Units['m/s']
    segment.climb_rate   = 1800. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)
    
     # ------------------------------------------------------------------
    #   Fourth Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_4"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 32000. * Units.ft
    segment.air_speed    = 230.0* Units['m/s']
    segment.climb_rate   = 900. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------
    #   Fifth Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_5"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 37000. * Units.ft
    segment.air_speed    = 230.0  * Units['m/s']
    segment.climb_rate   = 300.   * Units['ft/min']

    # add to mission
    mission.append_segment(segment)    
    
   
    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "cruise"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet

    segment.air_speed  = 450. * Units.knots
    segment.distance   = 2050. * Units.nmi

    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle  = ones_row(1) * 4.  * Units.degrees  

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 9.31  * Units.km
    segment.air_speed    = 440.0 * Units.knots
    segment.descent_rate = 2600. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 365.0 * Units.knots
    segment.descent_rate = 2300. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 250.0 * Units.knots
    segment.descent_rate = 1500. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    
    
    #------------------------------------------------------------------
    ###         Reserve mission
    #------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle
    # ------------------------------------------------------------------
    
    #segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "reserve_climb_1"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 15000.0 * Units.ft
    segment.climb_rate     = 1800.   * Units['ft/min']
    segment.mach_end       = 0.3
    segment.mach_start     = 0.2
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.throttle   = ones_row(1) * 1.0 
    segment.state.unknowns.body_angle  = ones_row(1) * 8. * Units.degrees 
    
    # add to misison
    mission.append_segment(segment)
    
    
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "reserve_cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.air_speed  = 96.66 * Units['m/s']
    segment.distance   = 100.0 * Units.nautical_mile    
    mission.append_segment(segment)
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle  = ones_row(1) * 2. * Units.degrees 
    
    # ------------------------------------------------------------------
    #   Loiter Segment: constant mach, constant time
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude_Loiter(base_segment)
    segment.tag = "reserve_loiter"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.mach = 0.5
    segment.time = 30.0 * Units.minutes
    
    mission.append_segment(segment)    
    
    
    # ------------------------------------------------------------------
    #   Fifth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------
    
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "reserve_descent_1_base"
    
    segment.analyses.extend( analyses.landing )
    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00
    
    
    segment.altitude_end = 0.0   * Units.km
    segment.descent_rate = 3.0   * Units['m/s']
    
    
    segment.mach_end    = 0.24
    segment.mach_start  = 0.3
    
    # append to mission
    mission.append_segment(segment)
    
    #------------------------------------------------------------------
    ###         Reserve mission completed
    #------------------------------------------------------------------
    
    return mission


def max_range_setup(analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    base_segment.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
    
    atmosphere=SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.takeoff )

    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.048 * Units.km
    segment.air_speed      = 138.0 * Units['m/s']
    segment.climb_rate     = 2900. * Units['ft/min']

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 168.0 * Units['m/s']
    segment.climb_rate   = 2500. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 25000. * Units.ft
    segment.air_speed    = 200.0  * Units['m/s']
    segment.climb_rate   = 1700. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)
    
     # ------------------------------------------------------------------
    #   Fourth Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_4"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 32000. * Units.ft
    segment.air_speed    = 225.0* Units['m/s']
    segment.climb_rate   = 800. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------
    #   Fifth Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_5"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 36999. * Units.ft
    segment.air_speed    = 230.0  * Units['m/s']
    segment.climb_rate   = 300.   * Units['ft/min']

    # add to mission
    mission.append_segment(segment)   
    
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "cruise"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )
    
    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet

    segment.air_speed  = 450. * Units.knots
    segment.distance   = 2050. * Units.nmi
     
    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle  = ones_row(1) * 4. * Units.degrees  
    
    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 9.31  * Units.km
    segment.air_speed    = 440.0 * Units.knots
    segment.descent_rate = 2600. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 365.0 * Units.knots
    segment.descent_rate = 2300. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 250.0 * Units.knots
    segment.descent_rate = 1500. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete    
    # ------------------------------------------------------------------ 
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle
    # ------------------------------------------------------------------ 
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "reserve_climb_1"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 15000.0 * Units.ft
    segment.climb_rate     = 1800.   * Units['ft/min']
    segment.mach_end       = 0.3
    segment.mach_start     = 0.2
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.throttle   = ones_row(1) * 1.0  
    segment.state.unknowns.body_angle  = ones_row(1) * 8. * Units.degrees 
    
    # add to misison
    mission.append_segment(segment) 
    
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "reserve_cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.air_speed  = 96.66 * Units['m/s']
    segment.distance   = 100.0 * Units.nautical_mile    
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle  = ones_row(1) * 2. * Units.degrees 
    
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Loiter Segment: constant mach, constant time
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude_Loiter(base_segment)
    segment.tag = "reserve_loiter"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.mach = 0.5
    segment.time = 30.0 * Units.minutes
    
    mission.append_segment(segment)    
    
    
    # ------------------------------------------------------------------
    #   Fifth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------
    
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "reserve_descent_1_max_range"
    
    segment.analyses.extend( analyses.landing )
    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00
    
    
    segment.altitude_end = 0.0   * Units.km
    segment.descent_rate = 3.0   * Units['m/s']
    
    
    segment.mach_end    = 0.24
    segment.mach_start  = 0.3
    
    # append to mission
    mission.append_segment(segment)
    
    #------------------------------------------------------------------
    ###         Reserve mission completed
    #------------------------------------------------------------------
    
    return mission


def short_field_setup(analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'the_mission'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()

    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    base_segment.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
    
    atmosphere=SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet = SUAVE.Attributes.Planets.Earth()
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Throttle
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.short_field_takeoff )

    # define segment attributes
    segment.atmosphere     = atmosphere
    segment.planet         = planet

    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 3.048 * Units.km
    segment.air_speed      = 138.0 * Units['m/s']
    segment.climb_rate     = 2900. * Units['ft/min']

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 168.0 * Units['m/s']
    segment.climb_rate   = 2500. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Speed, Constant Climb Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 25000. * Units.ft
    segment.air_speed    = 200.0  * Units['m/s']
    segment.climb_rate   = 1800. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)
    
     # ------------------------------------------------------------------
    #   Fourth Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_4"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 32000. * Units.ft
    segment.air_speed    = 230.0* Units['m/s']
    segment.climb_rate   = 900. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)   
    
    # ------------------------------------------------------------------
    #   Fifth Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate()
    segment.tag = "climb_5"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 37000. * Units.ft
    segment.air_speed    = 230.0  * Units['m/s']
    segment.climb_rate   = 300.   * Units['ft/min']

    # add to mission
    mission.append_segment(segment)    
        
    
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = "cruise"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere = atmosphere
    segment.planet     = planet

    segment.air_speed  = 450. * Units.knots
    segment.distance   = 2050. * Units.nmi

    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle  = ones_row(1) * 4. * Units.degrees  
    
    # add to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_1"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 9.31  * Units.km
    segment.air_speed    = 440.0 * Units.knots
    segment.descent_rate = 2600. * Units['ft/min']

    # add to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Second Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_2"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 3.657 * Units.km
    segment.air_speed    = 365.0 * Units.knots
    segment.descent_rate = 2300. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)


    # ------------------------------------------------------------------
    #   Third Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------

    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = "descent_3"

    # connect vehicle configuration
    segment.analyses.extend( analyses.cruise )

    # segment attributes
    segment.atmosphere   = atmosphere
    segment.planet       = planet

    segment.altitude_end = 0.0   * Units.km
    segment.air_speed    = 250.0 * Units.knots
    segment.descent_rate = 1500. * Units['ft/min']

    # append to mission
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Mission definition complete    
    # ------------------------------------------------------------------


    #------------------------------------------------------------------
    ###         Reserve mission
    #------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle
    # ------------------------------------------------------------------
    
    #segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment = Segments.Climb.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "reserve_climb_1"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 15000.0 * Units.ft
    segment.climb_rate     = 1800.   * Units['ft/min']
    segment.mach_end       = 0.3
    segment.mach_start     = 0.2
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.throttle   = ones_row(1) * 1.0 
    segment.state.unknowns.body_angle  = ones_row(1) * 8. * Units.degrees 
    
    # add to misison
    mission.append_segment(segment)
    
    
    
    # ------------------------------------------------------------------
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "reserve_cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.air_speed  = 96.66 * Units['m/s']
    segment.distance   = 100.0 * Units.nautical_mile    
    
    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle  = ones_row(1) * 2. * Units.degrees 
    
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   Loiter Segment: constant mach, constant time
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Mach_Constant_Altitude_Loiter(base_segment)
    segment.tag = "reserve_loiter"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.mach = 0.5
    segment.time = 30.0 * Units.minutes
    
    mission.append_segment(segment)    
    
    
    # ------------------------------------------------------------------
    #   Fifth Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------
    
    segment = Segments.Descent.Linear_Mach_Constant_Rate(base_segment)
    segment.tag = "reserve_descent_1_short_field"
    
    segment.analyses.extend( analyses.landing )
    analyses.landing.aerodynamics.settings.spoiler_drag_increment = 0.00
    
    
    segment.altitude_end = 0.0   * Units.km
    segment.descent_rate = 3.0   * Units['m/s']
    
    
    segment.mach_end    = 0.24
    segment.mach_start  = 0.3
    
    # append to mission
    mission.append_segment(segment)
    
    #------------------------------------------------------------------
    ###         Reserve mission completed
    #------------------------------------------------------------------
    
    return mission


def takeoff_mission_setup(analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission segment for takeoff
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'takeoff'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    base_segment.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
    
    # Climb Segment: Constant throttle, constant speed
    segment = Segments.Climb.Constant_Throttle_Constant_Speed(base_segment)
    segment.tag = "climb"    
    segment.analyses.extend( analyses.takeoff )
    segment.altitude_start =  35. *  Units.fts
    segment.altitude_end   = 304.8 *  Units.meter
    segment.air_speed      = 85.4 * Units['m/s']
    segment.throttle       = 1. 

    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle             = ones_row(1) * 12. * Units.deg  
    segment.state.unknowns.wind_angle             = ones_row(1) * 5. * Units.deg  
    
    mission.append_segment(segment)

    # Cutback Segment: Constant speed, constant segment angle
    segment = Segments.Climb.Constant_Speed_Constant_Angle_Noise(base_segment)
    segment.tag = "cutback"    
    segment.analyses.extend( analyses.takeoff )
    segment.air_speed    = 85.4 * Units['m/s']
    segment.climb_angle   = 2.86  * Units.degrees 
    mission.append_segment(segment)  
    
    return mission

def sideline_mission_setup(analyses):
    # ------------------------------------------------------------------
    #   Initialize the Mission segment for takeoff
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'sideline_takeoff'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    mission.airport = airport
    
    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    base_segment.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
    
    # Climb Segment: Constant throttle, constant speed
    segment = Segments.Climb.Constant_Throttle_Constant_Speed(base_segment)
    segment.tag = "climb"    
    segment.analyses.extend( analyses.takeoff )
    segment.altitude_start =  35. *  Units.fts
    segment.altitude_end   = 1600 *  Units.fts
    segment.air_speed      = 85.4 * Units['m/s']
    segment.throttle       = 1. 

    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle             = ones_row(1) * 12. * Units.deg  
    segment.state.unknowns.wind_angle             = ones_row(1) * 5. * Units.deg  
    
    mission.append_segment(segment)
    
    return mission    

def takeoff_mission_initialization(analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission segment for takeoff
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'takeoff_initialization'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()
    base_segment.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
    
    # Climb Segment: Constant speed, constant segment angle
    segment = Segments.Climb.Constant_Throttle_Constant_Speed(base_segment)
    segment.tag = "climb"    
    segment.analyses.extend( analyses.takeoff )
    segment.altitude_start =  35. *  Units.fts 
    segment.altitude_end   = 300. * Units.meter
    segment.air_speed      =  85.4 * Units['m/s']
    segment.throttle       = 1. 
     
    ones_row = segment.state.ones_row
    segment.state.unknowns.body_angle             = ones_row(1) * 12. * Units.deg  
    segment.state.unknowns.wind_angle             = ones_row(1) * 5. * Units.deg  
    
    mission.append_segment(segment)    
    
    return mission


def landing_mission_setup(analyses):
    
    # ------------------------------------------------------------------
    #   Initialize the Mission segment for landing
    # ------------------------------------------------------------------
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'landing'

    #airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    mission.airport = airport    

    # unpack Segments module
    Segments = SUAVE.Analyses.Mission.Segments

    # base segment
    base_segment = Segments.Segment()   
    base_segment.state.numerics.discretization_method = SUAVE.Methods.Utilities.Chebyshev.linear_data
    
    # ------------------------------------------------------------------
    #   Descent Segment: Constant speed, constant segment angle
    # ------------------------------------------------------------------
    segment = Segments.Descent.Constant_Speed_Constant_Angle_Noise(base_segment)
    segment.tag = "descent"
    segment.analyses.extend( analyses.landing )

    segment.air_speed    = 67. * Units['m/s']
    segment.descent_angle = 3.0   * Units.degrees  
    mission.append_segment(segment)
        
    return mission

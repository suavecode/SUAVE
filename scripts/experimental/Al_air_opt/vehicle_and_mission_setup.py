import SUAVE
from SUAVE.Core import Units



# ----------------------------------------------------------------------
#   Define vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
        
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'short_range_transport'    
    
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff               = 15000.   # kg
    vehicle.mass_properties.operating_empty           = 9000.   # kg
    vehicle.mass_properties.takeoff                   = 14500.   # kg
    #vehicle.mass_properties.max_zero_fuel             = 0.9 * vehicle.mass_properties.max_takeoff 
    vehicle.mass_properties.cargo                     = 0.  * Units.kilogram   
    
    vehicle.mass_properties.center_of_gravity         = [60 * Units.feet, 0, 0]  # Not correct
    vehicle.mass_properties.moments_of_inertia.tensor = [[10 ** 5, 0, 0],[0, 10 ** 6, 0,],[0,0, 10 ** 7]] # Not Correct
    
    # envelope properties
    vehicle.envelope.ultimate_load = 2.5 * 1.5
    vehicle.envelope.limit_load    = 2.5

    # basic parameters
    vehicle.reference_area         = 30.       
    vehicle.passengers             = 30
    vehicle.systems.control        = "fully powered" 
    vehicle.systems.accessories    = "medium range"
    
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.aspect_ratio            = 10.18
    wing.sweep                   = 0 * Units.deg
    wing.areas.reference         = 32.4779 
    wing.thickness_to_chord      = 0.1218 
    wing.taper                   = 0.2083 
    
    wing.span_efficiency         = 0.9  
   
    wing.flaps.type              = 'double_slotted'
    wing.flaps.chord             = 0.28        
    
    wing.areas.wetted            = 1.75 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.6 * wing.areas.reference
    
    wing.twists.root             = 3.5 * Units.degrees
    wing.twists.tip              = 3.0 * Units.degrees
    
    wing.origin                  = [20,0,0]
    wing.aerodynamic_center      = [3,0,0] 
    
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True
    
    wing.dynamic_pressure_ratio  = 1.0
    
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'
    
    wing.aspect_ratio            = 6.16      #
    wing.sweep                   = 0 * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.4
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 14.146      #

    wing.chords.root             = 3.28
    wing.chords.tip              = 0.763
    wing.chords.mean_aerodynamic = 1.696

    #wing.areas.reference         = 6.4593    #
    wing.areas.wetted            = 1.7 * wing.areas.reference
    wing.areas.exposed           = 0.8 * wing.areas.wetted
    wing.areas.affected          = 0.0 
    
    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 3.0 * Units.degrees  
    
    wing.origin                  = [34,0,0]
    wing.aerodynamic_center      = [2,0,0]
    
    wing.vertical                = False 
    wing.symmetric               = True
    
    wing.dynamic_pressure_ratio  = 0.9  
    
    # add to vehicle
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    
    wing.aspect_ratio            = 1.91      #
    wing.sweep                   = 0. * Units.deg
    wing.thickness_to_chord      = 0.11
    wing.taper                   = 0.25
    wing.span_efficiency         = 0.9
    
    wing.spans.projected         = 7.877      #    

    wing.chords.root             = 5.79
    wing.chords.tip              = 1.57
    wing.chords.mean_aerodynamic = 3.624
    
    #wing.areas.reference         = 26.4    #
    wing.areas.wetted            = 2.0 * wing.areas.reference
    wing.areas.exposed           = 0.95 * wing.areas.wetted
    wing.areas.affected          = 0.0 
    
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees  
    
    wing.origin                  = [31,0,0]
    wing.aerodynamic_center      = [2,0,0]    
    
    wing.vertical                = True 
    wing.symmetric               = False
    wing.t_tail                  = False
    
    wing.dynamic_pressure_ratio  = 1.0
  
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    #fuselage.number_coach_seats    = vehicle.passengers
    fuselage.seats_abreast         = 3.
    fuselage.number_coach_seats    =30.
    fuselage.seat_pitch            = 32.*Units.inches
    
    fuselage.fineness.nose         = 1.5
    fuselage.fineness.tail         = 2.
    
    fuselage.lengths.nose          = 3.45
    fuselage.lengths.tail          = 3.45/.75
    fuselage.lengths.cabin         = 9.5 # 28.85 #44.0
    fuselage.lengths.total         = fuselage.lengths.nose+\
    fuselage.lengths.tail+fuselage.lengths.cabin 
    fuselage.lengths.fore_space    = 6.
    fuselage.lengths.aft_space     = 5.    
    width                          =2.3
    fuselage.width                 = width
    
    fuselage.heights.maximum       = 2.3
    fuselage.heights.at_quarter_length          = width
    fuselage.heights.at_three_quarters_length   = width
    fuselage.heights.at_wing_root_quarter_chord = width

    fuselage.areas.side_projected  = 3.74* 38.02 #4.* 59.8 #  Not correct
    fuselage.areas.wetted          = 3.1415 * fuselage.heights.maximum * (fuselage.lengths.cabin + fuselage.lengths.nose * .8 + fuselage.lengths.tail * .8) # + 446.718 #688.64
    fuselage.areas.front_projected = 12.57
    
    fuselage.effective_diameter    = 3.74 #4.0
    
    fuselage.differential_pressure = 8.5 * Units.psi    # Maximum differential pressure
    
    # add to vehicle
    vehicle.append_component(fuselage)
    
    ################
    
    # ------------------------------------------------------------------
    #  Propulsion
    # ------------------------------------------------------------------
    max_alt=11*Units.km
    
    atm = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    p1, T1, rho1, a1, mew1 = atm.compute_values(0.)
    p2, T2, rho2, a2, mew2 = atm.compute_values(max_alt)
  
    
    sizing_segment = SUAVE.Components.Propulsors.Segments.Segment()
    sizing_segment.M   = 230./a2        
    sizing_segment.alt = max_alt
    sizing_segment.T   = T2           
    
    sizing_segment.p   = p2     
    #create battery
    battery = SUAVE.Components.Energy.Storages.Batteries.Variable_Mass.Aluminum_Air()
    #battery_lis = SUAVE.Components.Energy.Storages.Battery()
    #battery_lis.type='Li_S'
    #battery_lis.tag='Battery_Li_S'
    battery.tag = 'battery'
   
    # attributes
 
    #ducted fan
    ducted_fan= SUAVE.Components.Propulsors.Ducted_Fan_Bat()
    ducted_fan.tag='ducted_fan'
    #ducted_fan.propellant = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    ducted_fan.diffuser_pressure_ratio = 0.98
    ducted_fan.fan_pressure_ratio = 1.65
    ducted_fan.fan_nozzle_pressure_ratio = 0.99
    #ducted_fan.design_thrust = 2.5*Preq/V_cruise #factor of 2.5 accounts for top of climb
    ducted_fan.number_of_engines=4.0   
    ducted_fan.eta_pe=.95         #electric efficiency of battery
    ducted_fan.engine_sizing_ducted_fan(sizing_segment)   #calling the engine sizing method 
    vehicle.propulsor=ducted_fan    
   
    # ------------------------------------------------------------------
    #  Define the Energy Network
    # ------------------------------------------------------------------ 
    
    net=SUAVE.Components.Energy.Networks.Basic_Battery()
    net.propulsor=ducted_fan
    net.nacelle_diameter=ducted_fan.nacelle_diameter
    net.engine_length=ducted_fan.engine_length
    net.tag='network'
    net.append(ducted_fan)
    net.battery=battery
    net.number_of_engines=ducted_fan.number_of_engines
    
    vehicle.propulsors.append(net)
    # done!!
    return vehicle


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
    
    
    # ------------------------------------------------------------------
    #   First Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_1"
    
    segment.analyses.extend( analyses.takeoff )
    
    segment.altitude_start = 0.0   * Units.km
    segment.altitude_end   = 1.0   * Units.km
    segment.air_speed      = 80. * Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']
    
    # add to misison
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Second Climb Segment: constant Speed, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end   = 2.0   * Units.km
    segment.air_speed      = 80.* Units['m/s']
    segment.climb_rate     = 6.0   * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------
    #   Third Climb Segment: constant Mach, constant segment angle 
    # ------------------------------------------------------------------    
    
    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_3"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end = 3. * Units.km
    segment.air_speed    =140* Units['m/s']
    segment.climb_rate   = 3.0    * Units['m/s']
    
    # add to mission
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   Cruise Segment: constant speed, constant altitude
    # ------------------------------------------------------------------    
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.air_speed  = 147.1479 * Units['m/s']
    segment.distance   = 300 * Units.nautical_miles
        
    mission.append_segment(segment)
    
    
    # ------------------------------------------------------------------    
    #   First Descent Segment: consant speed, constant segment rate
    # ------------------------------------------------------------------    
    
    segment = Segments.Descent.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_1"
    
    segment.analyses.extend( analyses.cruise )
    
    segment.altitude_end = 2.   * Units.km
    segment.air_speed    = 120.0 * Units['m/s']
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
    segment.air_speed    = 100.0 * Units['m/s']
    segment.descent_rate = 5.0   * Units['m/s']

    # append to mission
    mission.append_segment(segment)
    
    # ------------------------------------------------------------------    
    #   Mission definition complete    
    # ------------------------------------------------------------------
    
    return mission
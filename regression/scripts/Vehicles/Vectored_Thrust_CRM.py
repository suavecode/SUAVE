import SUAVE
from SUAVE.Core import Units, Data
import copy
from SUAVE.Components.Energy.Networks.Vectored_Thrust import Vectored_Thrust
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass , compute_optimal_motor_parameters
from SUAVE.Methods.Propulsion import propeller_design 
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff 
from SUAVE.Methods.Weights.Buildups.Electric_Vectored_Thrust.empty import empty
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data

import numpy as np
import pylab as plt
from copy import deepcopy


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main(): 
    
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    
    configs.finalize()
    analyses.finalize()    
            
    
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate() 
    
    # plot results
    plot_mission(results)    
    
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


def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    #vehicle = SUAVE.Vehicle(SUAVE.Input_Output.SUAVE.load('cora_vehicle.res'))
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Vahana'
    vehicle.configuration = 'eVTOL'
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff         = 2250. * Units.lb 
    vehicle.mass_properties.operating_empty = 2250. * Units.lb
    vehicle.mass_properties.max_takeoff     = 2250. * Units.lb
    vehicle.mass_properties.center_of_gravity = [ 2.0144,   0.  ,  0.]
    
     
    # This needs updating
    # basic parameters
    vehicle.reference_area                    = 10.58275476  
    vehicle.envelope.ultimate_load            = 5.7
    vehicle.envelope.limit_load               = 3.    
  
    
    
    # ------------------------------------------------------				
    # WINGS				
    # ------------------------------------------------------				
    # WING PROPERTIES	
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag	                = 'main_wing'		
    wing.aspect_ratio	        = 11.37706641		
    wing.sweeps.quarter_chord	= 0.0
    wing.thickness_to_chord	= 0.18		
    wing.taper	                = 1.		
    wing.span_efficiency	= 0.9		
    wing.spans.projected	= 6.65	
    wing.chords.root	        = 0.95	
    wing.total_length	        = 0.95			
    wing.chords.tip	        = 0.95	
    wing.chords.mean_aerodynamic= 0.95			
    wing.dihedral	        = 0.0		
    wing.areas.reference	= 6.31		
    wing.areas.wetted	        = 12.635		
    wing.areas.exposed	        = 12.635		
    wing.twists.root	        = 0.		
    wing.twists.tip	        = 0.		
    wing.origin	                = [0.0,  0.0 , 0.0]
    wing.aerodynamic_center	= [0., 0., 0.]		   
    wing.winglet_fraction       = 0.0  
    wing.symmetric              = True
    
    # Segment 	
    segment = SUAVE.Components.Wings.Segment()
    segment.tag			= 'Section_1'		
    segment.origin		= [0., 0. , 0.]	
    segment.percent_span_location =	0.		
    segment.twist		= 0.		
    segment.root_chord_percent	= 1.		
    segment.dihedral_outboard	= 0.	
    segment.sweeps.quarter_chord= 0.	
    segment.thickness_to_chord	= 0.18		
    wing.Segments.append(segment)
    
   
    # Segment 	
    segment = SUAVE.Components.Wings.Segment()
    segment.tag			= 'Section_2'		
    segment.origin		= [0. , 0.,  0.]		
    segment.percent_span_location = 1.	
    segment.twist		= 0.		
    segment.root_chord_percent	= 1.	
    segment.dihedral_outboard	= 0.
    segment.sweeps.quarter_chord= 0.	
    segment.thickness_to_chord	= 0.18			
    wing.Segments.append(segment) 
    
    # add to vehicle
    vehicle.append_component(wing)       

    # WING PROPERTIES	
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag	        = 'main_wing_2'		
    wing.aspect_ratio	        = 11.37706641		
    wing.sweeps.quarter_chord	= 0.0
    wing.thickness_to_chord	= 0.18		
    wing.taper	                = 1.		
    wing.span_efficiency	= 0.9		
    wing.spans.projected	= 6.65	
    wing.chords.root	        = 0.95	
    wing.total_length	        = 0.95			
    wing.chords.tip	        = 0.95	
    wing.chords.mean_aerodynamic= 0.95			
    wing.dihedral	        = 0.0		
    wing.areas.reference	= 6.31		
    wing.areas.wetted	        = 12.635		
    wing.areas.exposed	        = 12.635		
    wing.twists.root	        = 0.		
    wing.twists.tip	        = 0.		
    wing.origin	                = [ 5.138, 0.0 ,1.24 ]
    wing.aerodynamic_center	= [0., 0., 0.]		   
    wing.winglet_fraction       = 0.0  
    wing.symmetric              = True
    
    # Segment 	
    segment = SUAVE.Components.Wings.Segment()
    segment.tag			= 'Section_1'		
    segment.origin		= [0., 0. , 0.]	
    segment.percent_span_location =	0.		
    segment.twist		= 0.		
    segment.root_chord_percent	= 1.		
    segment.dihedral_outboard	= 0.	
    segment.sweeps.quarter_chord= 0.	
    segment.thickness_to_chord	= 0.18		
    wing.Segments.append(segment)
    
   
    # Segment 	
    segment = SUAVE.Components.Wings.Segment()
    segment.tag			= 'Section_2'		
    segment.origin		= [0. , 0.,  0.]		
    segment.percent_span_location = 1.	
    segment.twist		= 0.		
    segment.root_chord_percent	= 1.	
    segment.dihedral_outboard	= 0.
    segment.sweeps.quarter_chord= 0.	
    segment.thickness_to_chord	= 0.18			
    wing.Segments.append(segment) 
    
    # add to vehicle 
    vehicle.append_component(wing)    
        
    # ------------------------------------------------------				
    # FUSELAGE				
    # ------------------------------------------------------				
    # FUSELAGE PROPERTIES
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                = 'fuselage'
    fuselage.configuration	= 'Tube_Wing'		
    fuselage.origin	        = [0. , 0.,  0.]	
    fuselage.seats_abreast	= 0.		
    fuselage.seat_pitch  	= 1.		
    fuselage.fineness.nose	= 1.5	
    fuselage.fineness.tail	= 4.0	
    fuselage.lengths.nose	= 1.7  	
    fuselage.lengths.tail	= 2.7	
    fuselage.lengths.cabin	= 1.7		
    fuselage.lengths.total	= 6.1		
    fuselage.width	        = 1.15 	
    fuselage.heights.maximum	=  1.7	
    fuselage.heights.at_quarter_length	        = 1.2		
    fuselage.heights.at_wing_root_quarter_chord	= 1.7		
    fuselage.heights.at_three_quarters_length	= 0.75	
    fuselage.areas.wetted	        = 12.97989862		
    fuselage.areas.front_projected	= 1.365211404		
    fuselage.effective_diameter 	= 1.318423736		
    fuselage.differential_pressure	= 0.		
    
    # Segment 	
    segment = SUAVE.Components.Fuselages.Segment() 
    segment.tag			= 'segment_1'		
    segment.origin	        = [0., 0. ,0.]		
    segment.percent_x_location	= 0.		
    segment.percent_z_location	= 0.		
    segment.height		= 0.		
    segment.width		= 0.		
    segment.length		= 0.		
    segment.effective_diameter	= 0.	
    fuselage.Segments.append(segment)  
                          
    # Segment 
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			= 'segment_2'		
    segment.origin		= [0., 0. ,0.]		
    segment.percent_x_location	= 0.275	
    segment.percent_z_location	= -0.009		
    segment.height		= 0.309*2		
    segment.width		= 0.28*2
    fuselage.Segments.append(segment)  
                          
    # Segment 
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			=' segment_3'		
    segment.origin		= [0., 0. ,0.]		
    segment.percent_x_location	= 0.768	
    segment.percent_z_location	= 0.046		
    segment.height		= 0.525*2	
    segment.width		= 0.445*2
    fuselage.Segments.append(segment)  
                          
    # Segment 	
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			= 'segment_4'		
    segment.origin		= [0., 0. ,0.]		
    segment.percent_x_location	= 0.25*6.2	
    segment.percent_z_location	= 0.209		
    segment.height		= 0.7*2		
    segment.width		= 0.55*2
    fuselage.Segments.append(segment)  
                          
    # Segment
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			= 'segment_5'		
    segment.origin		= [0., 0. ,0.]		
    segment.percent_x_location	= 0.5*6.2		
    segment.percent_z_location	= 0.407 		
    segment.height		= 0.850*2 		
    segment.width		= 0.61*2 
    segment.effective_diameter	= 0.		
    fuselage.Segments.append(segment)  
    
    
    # Segment 	
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			= 'segment_6'		
    segment.origin		= [0., 0. ,0.]		
    segment.percent_x_location	= 0.75 	
    segment.percent_z_location	= 0.771		
    segment.height		= 0.63*2	
    segment.width		= 0.442*2	
    fuselage.Segments.append(segment)  
                          
    # Segment
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			= 'segment_7'		
    segment.origin		= [0., 0. ,0.]		
    segment.percent_x_location	= 1.*6.2		
    segment.percent_z_location	= 1.192 		
    segment.height		= 0.165*2		
    segment.width		= 0.125*2	
    fuselage.Segments.append(segment)  
    
    # add to vehicle
    vehicle.append_component(fuselage)    
    
   
    #------------------------------------------------------------------
    # PROPULSOR
    #------------------------------------------------------------------
    net = Vectored_Thrust()    
    net.number_of_engines = 8
    net.thrust_angle      = 90.0 * Units.degrees #  conversion to radians, 
    net.nacelle_diameter  = 0.2921  # https://www.magicall.biz/products/integrated-motor-controller-magidrive/
    net.engine_length     = 0.106
    net.areas             = Data()
    net.areas.wetted      = np.pi*net.nacelle_diameter*net.engine_length + 0.5*np.pi*net.nacelle_diameter**2    
    net.voltage           = 500.

    
    #------------------------------------------------------------------
    # Design Electronic Speed Controller 
    #------------------------------------------------------------------
    esc                     = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency          = 0.95
    net.esc                 = esc

    #------------------------------------------------------------------
    # Design Payload
    #------------------------------------------------------------------
    payload                      = SUAVE.Components.Energy.Peripherals.Avionics()
    payload.power_draw           = 0.
    payload.mass_properties.mass = 200. * Units.kg
    net.payload                  = payload

    #------------------------------------------------------------------
    # Design Avionics
    #------------------------------------------------------------------
    avionics                     = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw          = 200. * Units.watts
    net.avionics                 = avionics

    #------------------------------------------------------------------
    # Design Battery
    #------------------------------------------------------------------
    bat                          = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.specific_energy          = 300. * Units.Wh/Units.kg
    bat.resistance               = 0.005
    bat.max_voltage              = net.voltage 
    bat.mass_properties.mass     = 350. * Units.kg
    initialize_from_mass(bat, bat.mass_properties.mass)
    net.battery                  = bat


    #------------------------------------------------------------------
    # Design Rotors and Propellers
    #------------------------------------------------------------------
    # atmosphere conditions 
    speed_of_sound                   = 340
    rho                              = 1.22 
    rad_per_sec_to_rpm               = 9.549
    
    fligth_CL = 0.75
    AR        = vehicle.wings.main_wing.aspect_ratio
    Cd0       = 0.06
    Cdi       = fligth_CL**2/(np.pi*AR*0.98)
    Cd        = Cd0 + Cdi 
    
    # ----------------------------------------------------------
    # PROPULSOR   
    # ----------------------------------------------------------
    # atmosphere conditions 
    speed_of_sound                   = 340
    rho                              = 1.22 
    rad_per_sec_to_rpm               = 9.549
    fligth_CL = 0.75
    AR        = vehicle.wings.main_wing.aspect_ratio
    Cd0       = 0.06
    Cdi       = fligth_CL**2/(np.pi*AR*0.98)
    Cd        = Cd0 + Cdi   
    
    # Create propeller geometry
    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                    = 'Vectored_Thrust_Propeller'    
    prop.y_pitch                = 1.850
    prop.tip_radius             = 0.8875  
    prop.hub_radius             = 0.1 
    prop.disc_area              = np.pi*(prop.tip_radius**2)   
    prop.design_tip_mach        = 0.5
    prop.number_blades          = 3  
    prop.freestream_velocity    = 85. * Units['ft/min'] # 110 mph         
    prop.angular_velocity       = prop.design_tip_mach* speed_of_sound* rad_per_sec_to_rpm /prop.tip_radius  * Units['rpm']      
    prop.design_Cl              = 0.7
    prop.design_altitude        = 500 * Units.feet                  
    Lift                        = vehicle.mass_properties.takeoff*9.81  
    prop.design_thrust          = (Lift * 1.5 )/net.number_of_engines 
    prop.induced_hover_velocity = np.sqrt(Lift/(2*rho*prop.disc_area*net.number_of_engines))  
    prop.design_power           = 0.0                         
    prop                        = propeller_design(prop)  
     
    # Front Propellers Locations
    prop_front                = Data()
    prop_front.origin         =  [[0.0 , 1.347 ,0.0 ]]
    prop_front.symmetric      = True
    prop_front.x_pitch_count  = 1
    prop_front.y_pitch_count  = 2     
    prop_front.y_pitch        = 1.85   
    
    # populating propellers on one side of wing
    if prop_front.y_pitch_count > 1 :
        for n in range(prop_front.y_pitch_count):
            if n == 0:
                continue
            for i in range(len(prop_front.origin)):
                proppeller_origin = [prop_front.origin[i][0] , prop_front.origin[i][1] +  n*prop_front.y_pitch ,prop_front.origin[i][2]]
                prop_front.origin.append(proppeller_origin)   
   
                 
    # populating proptellers on the other side of the vehicle   
    if prop_front.symmetric : 
        for n in range(len(prop_front.origin)):
            proppeller_origin = [prop_front.origin[n][0] , -prop_front.origin[n][1] ,prop_front.origin[n][2] ]
            prop_front.origin.append(proppeller_origin) 
      
    # Rear Propellers Locations 
    prop_rear = Data()
    prop_rear.origin              =  [[0.0 , 1.347 ,1.24 ]]  
    prop_rear.symmetric           = True
    prop_rear.x_pitch_count       = 1
    prop_rear.y_pitch_count       = 2     
    prop_rear.y_pitch             = 1.85                   
    # populating propellers on one side of wing
    if prop_rear.y_pitch_count > 1 :
        for n in range(prop_rear.y_pitch_count):
            if n == 0:
                continue
            for i in range(len(prop_rear.origin)):
                proppeller_origin = [prop_rear.origin[i][0] , prop_rear.origin[i][1] +  n*prop_rear.y_pitch ,prop_rear.origin[i][2]]
                prop_rear.origin.append(proppeller_origin)   


    # populating proptellers on the other side of thevehicle   
    if prop_rear.symmetric : 
        for n in range(len(prop_rear.origin)):
            proppeller_origin = [prop_rear.origin[n][0] , -prop_rear.origin[n][1] ,prop_rear.origin[n][2] ]
            prop_rear.origin.append(proppeller_origin) 
      
    # Assign all propellers (front and rear) to network
    prop.origin = prop_front.origin + prop_rear.origin   
    
    # append propellers to vehicle     
    net.propeller            = prop
    
    # Motor
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    # Propeller (Thrust) motor
    motor                      = SUAVE.Components.Energy.Converters.Motor()
    motor.mass_properties.mass = 9. * Units.kg
    motor.origin               = prop_front.origin + prop_rear.origin  
    motor.efficiency           = 0.935
    motor.gear_ratio           = 1. 
    motor.gearbox_efficiency   = 1. # Gear box efficiency        
    motor.nominal_voltage      = bat.max_voltage *3/4  
    motor.propeller_radius     = prop.tip_radius 
    motor.no_load_current      = 2.0 
    motor                      = compute_optimal_motor_parameters(motor,prop) 
    net.motor                  = motor 
    
    vehicle.append_component(net) 
     
    
    # Add extra drag sources from motors, props, and landing gear. All of these hand measured
    
    motor_height = .25 * Units.feet
    motor_width  =  1.6 * Units.feet
    
    propeller_width  = 1. * Units.inches
    propeller_height = propeller_width *.12
    
    main_gear_width  = 1.5 * Units.inches
    main_gear_length = 2.5 * Units.feet
    
    nose_gear_width  = 2. * Units.inches
    nose_gear_length = 2. * Units.feet
    
    nose_tire_height = (0.7 + 0.4) * Units.feet
    nose_tire_width  = 0.4 * Units.feet
    
    main_tire_height = (0.75 + 0.5) * Units.feet
    main_tire_width  = 4. * Units.inches
    
    total_excrescence_area_spin = 12.*motor_height*motor_width + \
        2.* main_gear_length*main_gear_width + nose_gear_width*nose_gear_length + \
        2 * main_tire_height*main_tire_width + nose_tire_height*nose_tire_width
    
    total_excrescence_area_no_spin = total_excrescence_area_spin + 12*propeller_height*propeller_width 
    
    vehicle.excrescence_area_no_spin = total_excrescence_area_no_spin 
    vehicle.excrescence_area_spin = total_excrescence_area_spin     
    
    
    # append motor origin spanwise locations onto wing data structure 
    motor_origins_front = np.array(prop_front.origin)
    vehicle.wings['main_wing'].motor_spanwise_locations = np.multiply(
        0.19 ,motor_origins_front[:,1])
    motor_origins_rear = np.array(prop_rear.origin)
    vehicle.wings['main_wing_2'].motor_spanwise_locations = np.multiply(
        0.19 ,motor_origins_rear[:,1])    
    
    # define weights analysis
    #vehicle.weight_breakdown = empty(vehicle)
    
    return vehicle

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
    weights = SUAVE.Analyses.Weights.Weights_Electric_Vectored_Thrust()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors 
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

    return analyses    


def mission_setup(analyses,vehicle):
        
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------

    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'mission'

    # airport
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
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.05 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)    
    
    
    # VSTALL Calculation
    m     = vehicle.mass_properties.max_takeoff
    g     = 9.81
    S     = vehicle.reference_area
    atmo  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho   = atmo.compute_values(1000.*Units.feet,0.).density
    CLmax = 1.2
    Vstall = float(np.sqrt(2.*m*g/(rho*S*CLmax)))
    
  
    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Hover.Climb(base_segment)
    segment.tag = "Departure"

    segment.analyses.extend( analyses.hover_climb ) 
    segment.altitude_start  = 0.0  * Units.ft
    segment.altitude_end    = 40.  * Units.ft
    segment.climb_rate      = 300. * Units['ft/min']
    segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy  
    
    segment.state.unknowns.propeller_power_coefficient = 0.04 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.8 * ones_row(1)
    
    segment.process.iterate.unknowns.network          = vehicle.propulsors.propulsor.unpack_unknowns_hover
    segment.process.iterate.residuals.network         = vehicle.propulsors.propulsor.residuals   
    segment.process.iterate.unknowns.mission          = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Hover Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Hover.Hover(base_segment)
    segment.tag = "Hover"

    segment.analyses.extend( analyses.hover )
 
    segment.altitude    = 40.  * Units.ft
    segment.time        = 2*60
    segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy

    segment.state.unknowns.propeller_power_coefficient      = 0.04 * ones_row(1)     
    segment.state.unknowns.throttle                         = 0.7 * ones_row(1)
    
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_hover
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals   
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)
      
    ## ------------------------------------------------------------------
    ##  First Transition Segment
    ## ------------------------------------------------------------------

    #segment =  Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    #segment.tag = "Transition_1"

    #segment.analyses.extend( analyses.transition_seg_1_4 )

    #segment.altitude        = 40.  * Units.ft
    #segment.air_speed_start = 300. * Units['ft/min']      #(~27kts )
    #segment.air_speed_end   = 35.  * Units['mph']       #(~75 kts, 38 m/s )
    #segment.acceleration    = 9.81/5
    #segment.pitch_initial   = 0  * Units.degrees
    #segment.pitch_final     = 7. * Units.degrees
           
    #segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    #segment.state.unknowns.throttle                    = 0.80 * ones_row(1)  

    #segment.process.iterate.unknowns.network        = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network       = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.unknowns.mission        = SUAVE.Methods.skip
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip

    ## add to misison
    #mission.append_segment(segment)
  
    ## ------------------------------------------------------------------
    ##  Second Transition Segment
    ## ------------------------------------------------------------------

    #segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    #segment.tag = "Transition_2"

    #segment.analyses.extend( analyses.transition_seg_2_3)

    #segment.altitude        = 40.  * Units.ft
    #segment.air_speed_start = 35.  * Units['mph'] 
    #segment.air_speed_end   = 85.  * Units['mph'] 
    #segment.acceleration    = 9.81/5
    #segment.pitch_initial   = 7. * Units.degrees
    #segment.pitch_final     = 5. * Units.degrees
           
    #segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    #segment.state.unknowns.throttle                    = 0.80 * ones_row(1) 

    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    #segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    ## add to misison
    #mission.append_segment(segment)
    
    # ------------------------------------------------------------------
    #   First Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "Climb"
    
    segment.analyses.extend(analyses.cruise)
    
    segment.climb_rate       = 600. * Units['ft/min']
    segment.air_speed_start  = 85.   * Units['mph']
    segment.air_speed_end    = 110.   * Units['mph']
    segment.altitude_start   = 40.0 * Units.ft
    segment.altitude_end     = 1000.0 * Units.ft               
    
    segment.state.unknowns.propeller_power_coefficient = 0.03 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.80 * ones_row(1)
    
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    # add to misison
    mission.append_segment(segment)     
                
    # ------------------------------------------------------------------
    #   First Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------
    
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "Cruise"
    
    segment.analyses.extend(analyses.cruise)
    
    segment.altitude  = 1000.0 * Units.ft
    segment.air_speed = 110.   * Units['mph']
    segment.distance  = 30.    * Units.miles                       
    
    segment.state.unknowns.propeller_power_coefficient = 0.03 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.60 * ones_row(1)
    
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    # add to misison
    mission.append_segment(segment)     
                
    # ------------------------------------------------------------------
    #   First Descent Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------
    
    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "Descent"
    
    segment.analyses.extend(analyses.cruise)
    segment.climb_rate       = -600. * Units['ft/min']
    segment.air_speed_start  = 110.   * Units['mph']
    segment.air_speed_end    = 85.   * Units['mph']
    segment.altitude_start   = 1000.0 * Units.ft
    segment.altitude_end     = 40.0 * Units.ft
    
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    # add to misison
    mission.append_segment(segment)     
    
  
    ## ------------------------------------------------------------------
    ##  Third Transition Segment
    ## ------------------------------------------------------------------

    #segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    #segment.tag = "Transition_3"

    #segment.analyses.extend( analyses.transition_seg_2_3)

    #segment.altitude        = 40.  * Units.ft
    #segment.air_speed_start = 85.  * Units['mph'] 
    #segment.air_speed_end   = 35.  * Units['mph'] 
    #segment.acceleration    = -9.81/5
    #segment.pitch_initial   = 0.0
    #segment.pitch_final     = 7. * Units.degrees
           
    #segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    #segment.state.unknowns.throttle                    = 0.60 * ones_row(1)     

    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    #segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    ## add to misison
    #mission.append_segment(segment)

    
    ## ------------------------------------------------------------------
    ##  Forth Transition Segment
    ## ------------------------------------------------------------------

    #segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    #segment.tag = "Transition_4"

    #segment.analyses.extend( analyses.transition_seg_1_4)

    #segment.altitude        = 40.  * Units.ft
    #segment.air_speed_start = 35. * Units['mph']       #(~27kts )
    #segment.air_speed_end   = 5 * Units['mph']      #(~75 kts, 38 m/s )
    #segment.acceleration    = -9.81/5
    #segment.pitch_initial   = 0.0
    #segment.pitch_final     = 7. * Units.degrees
           
    ##segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    ##segment.state.unknowns.throttle                    = 0.90 * ones_row(1)  

    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    #segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    ## add to misison
    #mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Descent Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Hover.Descent(base_segment)
    segment.tag = "Arrival"

    segment.analyses.extend( analyses.hover_descent )

    segment.altitude_start  = 40.0  * Units.ft
    segment.altitude_end    = 0.  * Units.ft
    segment.descent_rate    = 300. * Units['ft/min']
    
    #segment.state.unknowns.throttle                         = 0.5 * ones_row(1)
    #segment.state.unknowns.propeller_power_coefficient      = 0.02 * ones_row(1) 
    
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_hover
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals   
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

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
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):
    '''
    The configration set up below the scheduling of the nacelle angle and vehicle speed.
    Since one propeller operates at varying flight conditions, one must perscribe  the 
    pitch command of the propeller which us used in the variable pitch model in the analyses
    Note: low pitch at take off & low speeds, high pitch at cruise
    '''
    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Hover Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'hover'
    config.propulsors.propulsor.thrust_angle  = 90.0 * Units.degrees
    config.propulsors.propulsor.pitch_command = -5.  * Units.degrees    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Hover Climb Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'hover_climb'
    config.propulsors.propulsor.thrust_angle  = 90.0 * Units.degrees
    config.propulsors.propulsor.pitch_command = -5.  * Units.degrees    
    configs.append(config)

    # ------------------------------------------------------------------
    #   Hover-to-Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'transition_seg_1_4'
    config.propulsors.propulsor.thrust_angle     = 90.0  * Units.degrees
    config.propulsors.propulsor.pitch_command    = -5. * Units.degrees    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Hover-to-Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'transition_seg_2_3'
    config.propulsors.propulsor.thrust_angle     = 90.0  * Units.degrees  
    config.propulsors.propulsor.pitch_command    = -5. * Units.degrees  
    configs.append(config)
        
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    config.propulsors.propulsor.thrust_angle     =  0. * Units.degrees
    config.propulsors.propulsor.pitch_command    = 5.  * Units.degrees  
    configs.append(config)  
    
    # ------------------------------------------------------------------
    #   Hover Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'hover_descent'
    config.propulsors.propulsor.thrust_angle  = 90.0 * Units.degrees
    config.propulsors.propulsor.pitch_command = -5.  * Units.degrees    
    configs.append(config)
    
    return configs

# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results):
    plot_electronic_conditions(results)
    plot_proppeller_conditions(results)
    plot_eMotor_Prop_efficiencies(results)
    return    
 

# ------------------------------------------------------------------
#   Electronic Conditions
# ------------------------------------------------------------------
def plot_electronic_conditions(results, line_color = 'bo-', save_figure = False, save_filename = "Electronic_Conditions"):
    axis_font = {'size':'14'} 
    fig = plt.figure(save_filename)
    fig.set_size_inches(12, 10)
    for i in range(len(results.segments)):  
    
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        power    = results.segments[i].conditions.propulsion.battery_draw[:,0] 
        eta      = results.segments[i].conditions.propulsion.throttle[:,0]
        energy   = results.segments[i].conditions.propulsion.battery_energy[:,0] 
        volts    = results.segments[i].conditions.propulsion.voltage_under_load[:,0] 
        volts_oc = results.segments[i].conditions.propulsion.voltage_open_circuit[:,0]     
        current = results.segments[i].conditions.propulsion.current[:,0]      
        battery_amp_hr = (energy*0.000277778)/volts
        C_rating   = current/battery_amp_hr 
        
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, eta, 'bo-' ) 
        axes.set_ylabel('Throttle')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
        plt.ylim((0,1)) 
            
    
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, energy*0.000277778, line_color)
        axes.set_ylabel('Battery Energy (W-hr)',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, volts, 'bo-',label='Under Load')
        axes.plot(time,volts_oc, 'ks--',label='Open Circuit')
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Battery Voltage (Volts)',axis_font)  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')   
        if i == 0:
            axes.legend(loc='upper right')          
        axes.grid(True)         
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, C_rating, line_color)
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('C-Rating (C)',axis_font)  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)
 
    if save_figure:
        plt.savefig(save_filename + ".png")       
        
    return


# ------------------------------------------------------------------
#   Propulsion Conditions
# ------------------------------------------------------------------
def plot_proppeller_conditions(results, line_color = 'bo-', save_figure = False, save_filename = "Propeller"):
    axis_font = {'size':'14'} 
    fig = plt.figure(save_filename)
    fig.set_size_inches(12, 10)  
    for segment in results.segments.values(): 

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        rpm    = segment.conditions.propulsion.rpm[:,0] 
        thrust = segment.conditions.frames.body.thrust_force_vector[:,2]
        torque = segment.conditions.propulsion.motor_torque[:,0] 
        tm     = segment.conditions.propulsion.propeller_tip_mach[:,0]
 
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, -thrust, line_color)
        axes.set_ylabel('Thrust (N)',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
        
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, rpm, line_color)
        axes.set_ylabel('RPM',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)      
        
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, torque, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Torque (N-m)',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, tm, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Tip Mach',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)       
        
    if save_figure:
        plt.savefig(save_filename + ".png")  
            
    return

# ------------------------------------------------------------------
#   Electric Propulsion efficiencies
# ------------------------------------------------------------------
def plot_eMotor_Prop_efficiencies(results, line_color = 'bo-', save_figure = False, save_filename = "eMotor_Prop_Propulsor"):
    axis_font = {'size':'14'} 
    fig = plt.figure(save_filename)
    fig.set_size_inches(12, 10)  
    for segment in results.segments.values(): 

        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        effp   = segment.conditions.propulsion.etap[:,0]
        effm   = segment.conditions.propulsion.etam[:,0]
        
        axes = fig.add_subplot(1,2,1)
        axes.plot(time, effp, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Propeller Efficiency',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)           
        plt.ylim((0,1))
        
        axes = fig.add_subplot(1,2,2)
        axes.plot(time, effm, line_color )
        axes.set_xlabel('Time (mins)',axis_font)
        axes.set_ylabel('Motor Efficiency',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)
        
    if save_figure:
        plt.savefig(save_filename + ".png")  
            
    return
 

 
 

if __name__ == '__main__': 
    main()    
    plt.show()
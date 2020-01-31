# MultiRotor_CRM.py
# 
# Created: May 2019, M Clarke

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data
import copy
from SUAVE.Components.Energy.Networks.Vectored_Thrust import Vectored_Thrust
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Plots.Mission_Plots import *
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff 
from SUAVE.Methods.Weights.Buildups.Electric_Vectored_Thrust.empty import empty
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data

import numpy as np
import pylab as plt
from copy import deepcopy

import vsp 
from SUAVE.Input_Output.OpenVSP.vsp_write import write
from SUAVE.Input_Output.OpenVSP.get_vsp_areas import get_vsp_areas 
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    
    # configs.finalize()
    analyses.finalize()    
    
    # weight analysis
    #weights = analyses.configs.base.weights
    # breakdown = weights.evaluate()          
    
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # save results 
    save_results(results)    
    
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

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    vehicle               = SUAVE.Vehicle()
    vehicle.tag           = 'multicopter'
    vehicle.configuration = 'eVTOL'
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff           = 3000. * Units.lb 
    vehicle.mass_properties.operating_empty   = 2000. * Units.lb               # Approximate
    vehicle.mass_properties.max_takeoff       = 3000. * Units.lb               # Approximate
    vehicle.mass_properties.center_of_gravity = [8.5*0.3048 ,   0.  ,  0.*0.3048 ] # Approximate
    
     
    # This needs updating
    vehicle.passengers                        = 5
    vehicle.reference_area                    = 73  * Units.feet**2	
    vehicle.envelope.ultimate_load            = 5.7   
    vehicle.envelope.limit_load               = 3.  

    # ------------------------------------------------------				
    # FUSELAGE				
    # ------------------------------------------------------				
    # FUSELAGE PROPERTIES
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage'
    fuselage.configuration	                = 'Tube_Wing'		
    fuselage.origin	                        = [[0. , 0.,  0.]]	
    fuselage.seats_abreast	                = 2.		
    fuselage.seat_pitch  	                = 3.		
    fuselage.fineness.nose	                = 0.88 		
    fuselage.fineness.tail	                = 1.13 		
    fuselage.lengths.nose	                = 3.2   * Units.feet	
    fuselage.lengths.tail	                = 6.4 	* Units.feet
    fuselage.lengths.cabin	                = 6.4 	* Units.feet	
    fuselage.lengths.total	                = 16.0 	* Units.feet	
    fuselage.width	                        = 5.85  * Units.feet	
    fuselage.heights.maximum	                = 4.65  * Units.feet		
    fuselage.heights.at_quarter_length	        = 3.75  * Units.feet 	
    fuselage.heights.at_wing_root_quarter_chord	= 4.65  * Units.feet	
    fuselage.heights.at_three_quarters_length	= 4.26  * Units.feet	
    fuselage.areas.wetted	                = 236.  * Units.feet**2	
    fuselage.areas.front_projected	        = 0.14  * Units.feet**2	  	
    fuselage.effective_diameter 	        = 5.85  * Units.feet 	
    fuselage.differential_pressure	        = 0.	
    
    # Segment 	
    segment = SUAVE.Components.Fuselages.Segment() 
    segment.tag			                = 'segment_1'		
    segment.origin	                        = [0., 0. ,0.]		
    segment.percent_x_location	                = 0.		
    segment.percent_z_location	                = 0.0	
    segment.height		                = 0.1   * Units.feet 		
    segment.width		                = 0.1	* Units.feet 	 		
    segment.length		                = 0.		
    segment.effective_diameter	                = 0.1	* Units.feet 		
    fuselage.Segments.append(segment)  
                          
    # Segment 
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                = 'segment_2'		
    segment.origin		                = [4.*0.3048 , 0. ,0.1*0.3048 ] 	
    segment.percent_x_location	                = 0.25 	
    segment.percent_z_location	                = 0.05 
    segment.height		                = 3.75  * Units.feet 
    segment.width		                = 5.65  * Units.feet 	
    segment.length		                = 3.2   * Units.feet 	
    segment.effective_diameter	                = 5.65 	* Units.feet 
    fuselage.Segments.append(segment)  
                          
    # Segment 
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                =' segment_3'		
    segment.origin		                = [8.*0.3048 , 0. ,0.34*0.3048 ] 	
    segment.percent_x_location	                = 0.5 	
    segment.percent_z_location	                = 0.071 
    segment.height		                = 4.65  * Units.feet	
    segment.width		                = 5.55  * Units.feet 	
    segment.length		                = 3.2   * Units.feet
    segment.effective_diameter	                = 5.55  * Units.feet 
    fuselage.Segments.append(segment)  
                          
    # Segment 	
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                = 'segment_4'		
    segment.origin		                = [12.*0.3048 , 0. ,0.77*0.3048 ] 
    segment.percent_x_location	                = 0.75 
    segment.percent_z_location	                = 0.089 	
    segment.height		                = 4.73  * Units.feet		
    segment.width		                = 4.26  * Units.feet 		
    segment.length		                = 3.2   * Units.feet 	
    segment.effective_diameter	                = 4.26  * Units.feet 
    fuselage.Segments.append(segment)  
                          
    # Segment
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                = 'segment_5'		
    segment.origin		                = [16.*0.3048 , 0. ,2.02*0.3048 ] 
    segment.percent_x_location	                = 1.0
    segment.percent_z_location	                = 0.158 
    segment.height		                = 0.67	* Units.feet
    segment.width		                = 0.33	* Units.feet
    segment.length		                = 3.2   * Units.feet	
    segment.effective_diameter	                = 0.33  * Units.feet
    fuselage.Segments.append(segment)  
    

    fuselage.OpenVSP_values                      = Data() # These are incorrect !!    
    fuselage.OpenVSP_values.nose                 = Data()
    fuselage.OpenVSP_values.nose.top             = Data()
    fuselage.OpenVSP_values.nose.side            = Data()
    fuselage.OpenVSP_values.nose.top.angle       = 75.00 
    fuselage.OpenVSP_values.nose.top.strength    = 0.74 
    fuselage.OpenVSP_values.nose.side.angle      = 75.00 
    fuselage.OpenVSP_values.nose.side.strength   = 0.74  
    fuselage.OpenVSP_values.nose.TB_Sym          = True
    fuselage.OpenVSP_values.nose.z_pos           = 0.0 
    
    fuselage.OpenVSP_values.tail                 = Data()
    fuselage.OpenVSP_values.tail.top             = Data()
    fuselage.OpenVSP_values.tail.side            = Data()    
    fuselage.OpenVSP_values.tail.bottom          = Data()
    fuselage.OpenVSP_values.tail.top.angle       = -90.00 
    fuselage.OpenVSP_values.tail.top.strength    =  0.49 
    fuselage.OpenVSP_values.tail.side.angle      = -90.00  
    fuselage.OpenVSP_values.tail.side.strength   =  0.49   
    fuselage.OpenVSP_values.tail.TB_Sym          = True
    fuselage.OpenVSP_values.tail.bottom.angle    = -90.00   
    fuselage.OpenVSP_values.tail.bottom.strength = 0.49  
    fuselage.OpenVSP_values.tail.z_pos           = 0.158  
    
    # add to vehicle
    vehicle.append_component(fuselage)   
    
    #-------------------------------------------------------------------
    # BOOMS			
    #-------------------------------------------------------------------       
    boom_relative_rotation  = [30,90,150]
    boom = SUAVE.Components.Fuselages.Fuselage()
    boom.tag                                    = 'Boom_1R'
    boom.configuration	                        = 'Boom'
    boom.seats_abreast	                        = 0.		 	
    boom.seat_pitch	                        = 0.0	 
    boom.fineness.nose	                        = 0.	 		
    boom.fineness.tail	                        = 0.	 		
    boom.lengths.nose	                        = 0.5   * Units.feet			
    boom.lengths.tail	                        = 0.5   * Units.feet	 		
    boom.lengths.cabin	                        = 10.66 * Units.feet
    boom.lengths.total	                        = 11.66 * Units.feet			
    boom.width	                                = 0.5   * Units.feet			
    boom.heights.maximum                        = 0.5   * Units.feet			
    boom.heights.at_quarter_length	        = 0.5   * Units.feet	
    boom.heights.at_wing_root_quarter_chord     = 0.5   * Units.feet
    boom.heights.at_three_quarters_length	= 0.5   * Units.feet			
    boom.areas.wetted		                = 18.71 * Units.feet**2
    boom.areas.front_projected	                = 0.196 * Units.feet**2
    boom.effective_diameter	                = 0.5   * Units.feet	 		
    boom.differential_pressure	                = 0.
    boom.count                                  = 3
    boom.symmetric                              = True
    boom.no_lift                                = True 
    boom.z_rotation                             = boom_relative_rotation[0] 
    
    boom_center                                 = [8*0.3048 , 0., 3.5*0.3048 ]
    num_rotors                                  = 6
    offset                                      = (2*np.pi)/(2*num_rotors)
    angular_pitch                               = (2*np.pi)/num_rotors
    boom.origin	                                = [[boom_center[0] - boom.lengths.total*np.sin(offset),boom_center[1] - boom.lengths.total*np.cos(offset), boom_center[2] ]]   
    
    # add to vehicle
    vehicle.append_component(boom)    
    
    if boom.count >  1 : 
        for n in range(boom.count):
            if n == 0:
                continue
            else:
                index = n+1
                boom = deepcopy(vehicle.fuselages.boom_1r)
                boom.z_rotation = boom_relative_rotation[n]
                boom.origin     = [[boom_center[0] - boom.lengths.total*np.sin(offset + n*angular_pitch),boom_center[1] - boom.lengths.total*np.cos(offset + n*angular_pitch), boom_center[2] ]]   
                boom.tag = 'Boom_' + str(index) + 'R'
                vehicle.append_component(boom)    
    if boom.symmetric : 
        for n in range(boom.count):
            index = n+1
            boom = deepcopy(vehicle.fuselages.boom_1r)
            boom.z_rotation = -boom_relative_rotation[n] 
            boom.origin     = [[boom_center[0] - boom.lengths.total*np.sin(offset + n*angular_pitch),-(boom_center[1] - boom.lengths.total*np.cos(offset + n*angular_pitch)), boom_center[2] ]]   
            boom.tag = 'Boom_' + str(index) + 'L'
            vehicle.append_component(boom)       
    #------------------------------------------------------------------
    # PROPULSOR
    #------------------------------------------------------------------
    net = Vectored_Thrust()
    net.number_of_engines         = 6
    net.thrust_angle              = 90. * Units.degrees
    net.nacelle_diameter          = 0.6 * Units.feet	# need to check	
    net.engine_length             = 0.5 * Units.feet
    net.areas                     = Data()
    net.areas.wetted              = np.pi*net.nacelle_diameter*net.engine_length + 0.5*np.pi*net.nacelle_diameter**2    
    net.voltage                   =  500.

    #------------------------------------------------------------------
    # Design Electronic Speed Controller 
    #------------------------------------------------------------------
    esc                          = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency               = 0.95
    net.esc                      = esc
    
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
    bat.mass_properties.mass     = 300. * Units.kg
    initialize_from_mass(bat, bat.mass_properties.mass)
    net.battery                  = bat

    #------------------------------------------------------------------
    # Design Rotors and Propellers
    #------------------------------------------------------------------
    # Thrust Propeller
    prop                         = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                     = 'Propeller'
    prop.tip_radius              = 3.95  * Units.feet
    prop.hub_radius              = 0.6  * Units.feet
    prop.number_blades           = 3
    prop.freestream_velocity     = 10.   * Units['mph']  
    
    tip_speed                    = 0.5*343.
    prop.angular_velocity        = tip_speed*prop.tip_radius   
    prop.design_Cl               = 0.8
    prop.design_altitude         = 1000 * Units.feet                   
    prop.design_thrust           = (vehicle.mass_properties.takeoff*9.81/net.number_of_engines)*1.1    
    prop.design_power            = 0.0     
    prop                         = propeller_design(prop)  

    vehicle_weight               = vehicle.mass_properties.takeoff*9.81    
    rho                          = 1.225
    A                            = np.pi*(prop.tip_radius**2)
    prop.induced_hover_velocity  = np.sqrt(vehicle_weight/(2*rho*A*net.number_of_engines))
    
    prop.origin                  = []
    
    # propulating propellers on the other side of the vehicle    
    for fuselage in vehicle.fuselages:
        if fuselage.tag == 'fuselage':
            continue
        else:
            prop.origin.append(fuselage.origin[0])           
   
    # append propellers to vehicle           
    net.propeller                     = prop
    
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    # Motor
    motor = SUAVE.Components.Energy.Converters.Motor()
    motor.mass_properties.mass = 4. * Units.kg
    motor.origin               = prop.origin 
    motor.speed_constant       = 8.5 * Units.rpm   # 15.0 * Units['rpm/volt'] # 1.05
    motor.propeller_radius     = prop.tip_radius
    motor.propeller_Cp         = prop.Cp    
    motor.resistance           = .2  
    motor.no_load_current      = 5.    
    motor.gear_ratio           = 0.95
    motor.gearbox_efficiency   = 0.98
    motor.propeller_radius     = prop.tip_radius
    net.motor                  = motor

    # append motor origin spanwise locations onto wing data structure 
    motor_origins = np.array(prop.origin)
    #for wing in vehicle.wings:
        #if wing.tag == 'main_wing':
            #vehicle.wings[wing.tag].motor_spanwise_locations = np.multiply( 0.19,motor_origins[:,1]) #0.19
            
    vehicle.append_component(net)
    
    #----------------------------------------------------------------------------------------
    # Add extra drag sources from motors, props, and landing gear. All of these hand measured
    #----------------------------------------------------------------------------------------
    motor_height                            = .25 * Units.feet
    motor_width                             = 1.6 * Units.feet    
    propeller_width                         = 1. * Units.inches
    propeller_height                        = propeller_width *.12    
    main_gear_width                         = 1.5 * Units.inches
    main_gear_length                        = 2.5 * Units.feet    
    nose_gear_width                         = 2. * Units.inches
    nose_gear_length                        = 2. * Units.feet    
    nose_tire_height                        = (0.7 + 0.4) * Units.feet
    nose_tire_width                         = 0.4 * Units.feet    
    main_tire_height                        = (0.75 + 0.5) * Units.feet
    main_tire_width                         = 4. * Units.inches    
    total_excrescence_area_spin             = 12.*motor_height*motor_width + \
        2.*main_gear_length*main_gear_width + nose_gear_width*nose_gear_length + \
        2*main_tire_height*main_tire_width + nose_tire_height*nose_tire_width
    
    total_excrescence_area_no_spin          = total_excrescence_area_spin + 12*propeller_height*propeller_width 
                                           
    vehicle.excrescence_area_no_spin        = total_excrescence_area_no_spin 
    vehicle.excrescence_area_spin           = total_excrescence_area_spin     
    
    #----------------------------------------------------------------------------------------
    # EVALUATE WEIGHTS using calculation (battery not updated) 
    #----------------------------------------------------------------------------------------
    #vehicle.weight_breakdown                = empty(vehicle)
    #MTOW                                    = vehicle.weight_breakdown.total
    #Payload                                 = vehicle.weight_breakdown.payload
    #OE                                      = MTOW - vehicle.weight_breakdown.battery               
    #vehicle.mass_properties.takeoff         = MTOW
    #vehicle.mass_properties.operating_empty = OE  
    
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
    weights = SUAVE.Analyses.Weights.Weights_Electric_Helicopter()
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
    base_segment.state.numerics.number_control_points        = 10
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery 
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.05 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = 400. * ones_row(1)  # vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)     
    base_segment.state.unknowns.thurst_angle                 = 90. * Units.degrees * ones_row(1)
    base_segment.state.residuals.network                     = 0. * ones_row(3)    
    
    #base_segment.process.iterate.unknowns.mission            = SUAVE.Methods.skip
    #base_segment.process.iterate.conditions.stability        = SUAVE.Methods.skip
    #base_segment.process.finalize.post_process.stability     = SUAVE.Methods.skip    
    #base_segment.process.compute.lift.compressible_wings     = SUAVE.Methods.skip   
    #base_segment.process.compute.drag.induced                = SUAVE.Methods.skip   
    
    # VSTALL Calculation
    m     = vehicle.mass_properties.max_takeoff
    g     = 9.81
    S     = vehicle.reference_area
    atmo  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho   = atmo.compute_values(1000.*Units.feet,0.).density
    CLmax = 1.2
    
    
    Vstall = float(np.sqrt(2.*m*g/(rho*S*CLmax)))

    # ------------------------------------------------------------------
    #   First Taxi Segment: Constant Speed
    # ------------------------------------------------------------------

    #segment = Segments.Hover.Climb(base_segment)
    #segment.tag = "Ground_Taxi"

    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Hover.Climb(base_segment)
    segment.tag = "Climb_1"

    segment.analyses.extend( analyses.climb_1 )

    segment.altitude_start  = 0.0  * Units.ft
    segment.altitude_end    = 40.  * Units.ft
    segment.climb_rate      = 300. * Units['ft/min']
    segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy*0.95
    
    segment.state.unknowns.throttle                       = 0.75 * ones_row(1)
    segment.state.unknowns.propeller_power_coefficient    = 0.5 * ones_row(1) 

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip
    
    # add to misison
    mission.append_segment(segment)

    ## ------------------------------------------------------------------
    ##   Second Climb Segment: Constant Speed, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Climb_2"
    
    #segment.analyses.extend( analyses.climb_2  )
    
    #segment.altitude_start  = 40.0 * Units.ft
    #segment.altitude_end    = 300. * Units.ft
    #segment.climb_rate      = 500.  * Units['ft/min']
    #segment.air_speed_start = 1.2 * Vstall 
    #segment.air_speed_end   = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)    
    
    #segment.state.unknowns.throttle                       = 1.1 * ones_row(1) 
    
    #segment.process.iterate.unknowns.network        = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network       = vehicle.propulsors.propulsor.residuals   
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    ## add to misison
    #mission.append_segment(segment) 
    
    ## ------------------------------------------------------------------
    ##   Maneuver Segment: Constant Speed, Constant Altitude
    ## ------------------------------------------------------------------

    #segment = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    #segment.tag = "Departure_Terminal_Procedures"

    #segment.analyses.extend(analyses.departure_terminal_procedures )

    #segment.altitude  = 300.0 * Units.ft
    #segment.time      = 60.   * Units.second
    #segment.air_speed = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2) # 1.2*Vstall  
    
    #segment.state.unknowns.throttle                    = 0.5 * ones_row(1) 
    ##segment.state.unknowns.propeller_power_coefficient = 0.02 * ones_row(1)    
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals   
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip  
    
    ## add to misison
    #mission.append_segment(segment)
    
    ## ------------------------------------------------------------------
    ##   Third Climb Segment: Constant Acceleration, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Accelerated_Climb"
    
    #segment.analyses.extend( analyses.accelerated_climb )
    
    #segment.altitude_start  = 300.0 * Units.ft
    #segment.altitude_end    = 1000. * Units.ft
    #segment.climb_rate      = 500.  * Units['ft/min']
    #segment.air_speed_start = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    #segment.air_speed_end   = 110.  * Units['mph']                    
    
    ##segment.state.thrust_angle                             = 25. * Units.degrees 
    #segment.state.unknowns.throttle                        = 0.5 * ones_row(1) 
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    ## add to misison
    #mission.append_segment(segment)    
    
    ## ------------------------------------------------------------------
    ##   First Cruise Segment: Constant Acceleration, Constant Altitude
    ## ------------------------------------------------------------------
    
    #segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    #segment.tag = "Cruise"
    
    #segment.analyses.extend(analyses.cruise)
    
    #segment.altitude  = 1000.0 * Units.ft
    #segment.air_speed = 110.   * Units['mph']
    #segment.distance  = 60.    * Units.miles                       
    
    #segment.state.unknowns.throttle  = 0.5 * ones_row(1)
    ##segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    ## add to misison
    #mission.append_segment(segment)     
    
    ## ------------------------------------------------------------------
    ##   First Descent Segment: Constant Acceleration, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Decelerated_Descent"
    
    #segment.analyses.extend(analyses.decelerated_descent)  
    #segment.altitude_start  = 1000.0 * Units.ft
    #segment.altitude_end    = 300. * Units.ft
    #segment.climb_rate      = -500.  * Units['ft/min']
    #segment.air_speed_start = 110.   * Units['mph']
    #segment.air_speed_end   = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2) # 1.2*Vstall
    
    #segment.state.unknowns.throttle  = 0.6 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip 
    
    ## add to misison
    #mission.append_segment(segment)        
    
    ## ------------------------------------------------------------------
    ##   Maneuver Segment: Constant Speed, Constant Altitude
    ## ------------------------------------------------------------------
    
    #segment = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    #segment.tag = "Arrival_Terminal_Procedures"
    
    #segment.analyses.extend(analyses.arrival_terminal_procedures)
    
    #segment.altitude        = 300. * Units.ft
    #segment.air_speed       = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2) # 1.2*Vstall
    #segment.time            = 60 * Units.seconds
    
    #segment.state.unknowns.throttle  = 0.5 * ones_row(1) 
    ##segment.state.unknowns.propeller_power_coefficient = 0.06 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip   
    
    ## add to misison
    #mission.append_segment(segment)    
    
    ## ------------------------------------------------------------------
    ##   Second Descent Segment: Constant Speed, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Descent_1"
    
    #segment.analyses.extend(analyses.descent_1)
    
    #segment.altitude_start  = 300.0 * Units.ft
    #segment.altitude_end    = 40. * Units.ft
    #segment.climb_rate      = -500.  * Units['ft/min']
    #segment.air_speed_start = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    #segment.air_speed_end   = 1.2*Vstall                           
    
    #segment.state.unknowns.throttle  = 0.5 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip  
        
    
    ## add to misison
    #mission.append_segment(segment)       
    
    ## ------------------------------------------------------------------
    ##   Third Descent Segment: Constant Speed, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Hover.Descent(base_segment)
    #segment.tag = "Descent_2"
    
    #segment.analyses.extend(analyses.descent_2)
    
    #segment.altitude_start  = 40.0  * Units.ft
    #segment.altitude_end    = 10.   * Units.ft
    #segment.descent_rate    = 300. * Units['ft/min']
    #segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy
        
    ##segment.state.thrust_angle       = 90. * Units.degrees 
    #segment.state.unknowns.throttle  = 0.9 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals  
    #segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    #segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    
    ## add to misison
    #mission.append_segment(segment)    
    
            
    ## ------------------------------------------------------------------
    ##   RESERVE MISSION
    ## ------------------------------------------------------------------
  
    ## ------------------------------------------------------------------
    ##   First Climb Segment: Constant Speed, Constant Rate
    ## ------------------------------------------------------------------

    #segment = Segments.Hover.Climb(base_segment)
    #segment.tag = "Reserve_Climb_1"

    #segment.analyses.extend( analyses )

    #segment.altitude_start  = 0.0  * Units.ft
    #segment.altitude_end    = 40.  * Units.ft
    #segment.climb_rate      = 300. * Units['ft/min']
    #segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy 
    
    #segment.state.thrust_angle                            = 90. * Units.degrees 
    #segment.state.unknowns.throttle                       = 0.5 * ones_row(1)
    #segment.state.unknowns.propeller_power_coefficient    = 0.1 * ones_row(1) 

    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals
    #segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    #segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability   = SUAVE.Methods.skip
    
    ## add to misison
    #mission.append_segment(segment)

    ## ------------------------------------------------------------------
    ##   Second Climb Segment: Constant Speed, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Reserve_Climb_2"
    
    #segment.analyses.extend( analyses )
    
    #segment.altitude_start  = 40.0 * Units.ft
    #segment.altitude_end    = 300. * Units.ft
    #segment.climb_rate      = 500.  * Units['ft/min']
    #segment.air_speed_start = 1.2 * Vstall 
    #segment.air_speed_end   = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)    
    
    #segment.state.thrust_angle                            =  75. * Units.degrees 
    #segment.state.unknowns.throttle                       = 0.5 * ones_row(1) 
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals   
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    ## add to misison
    #mission.append_segment(segment) 
    
        
    ## ------------------------------------------------------------------
    ##   Third Climb Segment: Constant Acceleration, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Reserve_Accelerated_Climb"
    
    #segment.analyses.extend( analyses )
    
    #segment.altitude_start  = 300.0 * Units.ft
    #segment.altitude_end    = 1000. * Units.ft
    #segment.climb_rate      = 500.  * Units['ft/min']
    #segment.air_speed_start = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    #segment.air_speed_end   = 110.  * Units['mph']                    
    
    #segment.state.thrust_angle                             = 65. * Units.degrees 
    #segment.state.unknowns.throttle                        = 0.5 * ones_row(1) 
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    ## add to misison
    #mission.append_segment(segment)    
    
    ## ------------------------------------------------------------------
    ##   First Cruise Segment: Constant Acceleration, Constant Altitude
    ## ------------------------------------------------------------------
    
    #segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    #segment.tag = "Reserve_Cruise"
    
    #segment.analyses.extend(analyses)
    
    #segment.altitude  = 1000.0 * Units.ft
    #segment.air_speed = 110.   * Units['mph']
    #segment.distance  = 6.    * Units.miles                       
    
    #segment.state.unknowns.body_angle = ones_row(1) * 1. * Units.degrees 
    #segment.state.thrust_angle       = 5. * Units.degrees 
    #segment.state.unknowns.throttle  = 0.45 * ones_row(1) 
    #segment.state.unknowns.propeller_power_coefficient = 0.00075 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    ## add to misison
    #mission.append_segment(segment)     
    
    ## ------------------------------------------------------------------
    ##   First Descent Segment: Constant Acceleration, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Reserve_Decelerated_Descent"
    
    #segment.analyses.extend(analyses)  
    #segment.altitude_start  = 1000.0 * Units.ft
    #segment.altitude_end    = 300. * Units.ft
    #segment.climb_rate      = -500.  * Units['ft/min']
    #segment.air_speed_start = 110.   * Units['mph']
    #segment.air_speed_end   = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2) # 1.2*Vstall
    
    #segment.state.thrust_angle    = 35. * Units.degrees 
    #segment.state.unknowns.throttle  = 0.6 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip 
    
    ## add to misison
    #mission.append_segment(segment)        
        
    ## ------------------------------------------------------------------
    ##   Second Descent Segment: Constant Speed, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    #segment.tag = "Reserve_Descent_2"
    
    #segment.analyses.extend(analyses)
    
    #segment.altitude_start  = 300.0 * Units.ft
    #segment.altitude_end    = 40. * Units.ft
    #segment.climb_rate      = -500.  * Units['ft/min']
    #segment.air_speed_start = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    #segment.air_speed_end   = 1.2*Vstall                           
    
    #segment.state.thrust_angle    = 15. * Units.degrees 
    #segment.state.unknowns.throttle  = 0.6 * ones_row(1)
     
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    #segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability = SUAVE.Methods.skip  
        
    
    ## add to misison
    #mission.append_segment(segment)       
    
    ## ------------------------------------------------------------------
    ##   Third Descent Segment: Constant Speed, Constant Rate
    ## ------------------------------------------------------------------
    
    #segment = Segments.Hover.Descent(base_segment)
    #segment.tag = "Reserve_Descent_1"
    
    #segment.analyses.extend(analyses)
    
    #segment.altitude_start  = 40.0  * Units.ft
    #segment.altitude_end    = 10.   * Units.ft
    #segment.descent_rate    = 300. * Units['ft/min']
    #segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy
        
    #segment.state.thrust_angle    = 90. * Units.degrees 
    #segment.state.unknowns.throttle  = 0.6 * ones_row(1)
    
    #segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    #segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals  
    #segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    #segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    #segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    
    ## add to misison
    #mission.append_segment(segment)    
    
            
    ## ------------------------------------------------------------------
    ##   Mission definition complete    
    ## ------------------------------------------------------------------  
   
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
    config.propulsors.propulsor.pitch_command = 0.  * Units.degrees    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #    Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'climb_1'
    config.propulsors.propulsor.pitch_command = 0.  * Units.degrees    
    configs.append(config)

    # ------------------------------------------------------------------
    #     Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'climb_2'
    config.propulsors.propulsor.pitch_command  = -5.0 * Units.degrees  
    configs.append(config)
    
    # ------------------------------------------------------------------
    #     Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'departure_terminal_procedures'
    config.propulsors.propulsor.pitch_command = -5. * Units.degrees  
    configs.append(config)

    # ------------------------------------------------------------------
    #    Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'accelerated_climb'
    config.propulsors.propulsor.pitch_command    = 0  * Units.degrees  
    configs.append(config)  
    
        
    # ------------------------------------------------------------------
    #     Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    config.propulsors.propulsor.pitch_command    = 0  * Units.degrees  
    configs.append(config)  
    
    # ------------------------------------------------------------------
    #     Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'decelerated_descent'
    config.propulsors.propulsor.pitch_command = 0.  * Units.degrees    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #     Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'arrival_terminal_procedures' 
    config.propulsors.propulsor.pitch_command = -5. * Units.degrees   
    configs.append(config)
    
    # ------------------------------------------------------------------
    #     Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'descent_1'
    config.propulsors.propulsor.pitch_command = -10.  * Units.degrees    
    configs.append(config)    
    
    # ------------------------------------------------------------------
    #     Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'descent_2'
    config.propulsors.propulsor.pitch_command = -10.  * Units.degrees    
    configs.append(config)    
    
    
    return configs


# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results):
    plot_aerodynamic_coefficients(results)
    plot_disc_power_loading(results)
    plot_electronic_conditions(results)
    plot_flight_conditions(results)
    plot_propulsor_conditions(results)
    return    

def save_results(results):

    seg_count        = len(results.segments)
    saved_variables  = 23
    mission_segments = len(results.segments)
    control_points   = len(results.segments[0].conditions.frames.inertial.time[:,0])
    result_mat       = np.zeros((mission_segments*control_points,saved_variables))
    j = 0
    for segment in results.segments.values():  
        for i in range(control_points):
            time           = segment.conditions.frames.inertial.time[i,0] / Units.min
                           
            CLift          = segment.conditions.aerodynamics.lift_coefficient[i,0]
            CDrag          = segment.conditions.aerodynamics.drag_coefficient[i,0]
            AoA            = segment.conditions.aerodynamics.angle_of_attack[i,0] / Units.deg
            if abs(AoA) > 80:
                AoA = 0       
            l_d            = CLift/CDrag
               
            eta            = segment.conditions.propulsion.throttle[i,0]
            energy         = segment.conditions.propulsion.battery_energy[i,0] 
            volts          = segment.conditions.propulsion.voltage_under_load[i,0]
            volts_oc       = segment.conditions.propulsion.voltage_open_circuit[i,0]    
            current        = segment.conditions.propulsion.current[i,0]      
            battery_amp_hr = (energy*0.000277778)/volts
            C_rating       = current/battery_amp_hr 
            
            rpm            = segment.conditions.propulsion.rpm [i,0]
            rps            =  rpm/60
            thrust         = segment.conditions.frames.body.thrust_force_vector[i,0]
            torque         = segment.conditions.propulsion.motor_torque[i,0]   
            spec_power_in_bat   = segment.conditions.propulsion.battery_specfic_power[i,0] 
            power_in_shaft = torque/(2*np.pi*rps)*0.00134102                             # converting W to hp   
            power_out      = thrust*segment.conditions.freestream.velocity[i,0]*0.00134102    # converting W to hp   
            effp           = segment.conditions.propulsion.etap[i,0]
            effm           = segment.conditions.propulsion.etam[i,0]
            prop_omega     = segment.conditions.propulsion.rpm[i,0]*0.104719755  
            tip_mach       = segment.conditions.propulsion.propeller_tip_mach[i,0]
            V_inf          =  segment.conditions.freestream.velocity[i,0]
            result_mat[j]  = np.array([time,CLift,CDrag,l_d,AoA,eta,energy,volts,volts_oc,current ,
                                      battery_amp_hr, C_rating, rpm,thrust,torque, spec_power_in_bat, power_in_shaft ,
                                      power_out , effp , effm , prop_omega ,tip_mach ,V_inf])
            j += 1                                                                 
    np.save('Mulktirotor_Results.npy',result_mat)
    return        

if __name__ == '__main__': 
    main()    
    plt.show()
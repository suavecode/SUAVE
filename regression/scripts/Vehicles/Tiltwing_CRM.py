# Tiltwing_eVTOL.py
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
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff 
from SUAVE.Methods.Weights.Buildups.Electric_Vectored_Thrust.empty import empty
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data

import numpy as np
import pylab as plt
from copy import deepcopy

import vsp 
from SUAVE.Input_Output.OpenVSP.vsp_write import write
#from SUAVE.Input_Output.OpenVSP.get_vsp_areas import get_vsp_areas 
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    
    # configs.finalize()
    analyses.finalize()    
    
    # weight analysis
    weights =  analyses.configs.base.weights
    # breakdown = weights.evaluate()          
    
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()
        
    # plot results
    plot_mission(results,configs)
    
    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def full_setup():
    
    # vehicle data
    vehicle  = vehicle_setup()
    write(vehicle, "Tiltwing_CRM")
    configs  = configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = analyses_setup(configs)

    # mission analyses
    mission  = mission_setup(configs_analyses,vehicle)
    missions_analyses = missions_setup(mission,vehicle)
    
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
    vehicle.tag           = 'Tiltwing_CRM'
    vehicle.configuration = 'eVTOL'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff           = 3000. * Units.lb        # Approximate 
    vehicle.mass_properties.operating_empty   = 2000. * Units.lb        # Approximate
    vehicle.mass_properties.max_takeoff       = 3000. * Units.lb        # Approximate
    vehicle.mass_properties.center_of_gravity = [8.5*0.3048 ,   0.  ,  0.]  
    
    # basic parameters
    vehicle.passengers                        = 5
    vehicle.reference_area                    = 10 * Units.feet**2	
    vehicle.envelope.ultimate_load            = 5.7   
    vehicle.envelope.limit_load               = 3.  
    
    # ------------------------------------------------------				
    # WINGS				
    # ------------------------------------------------------				
    # Main Wing
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag	                  = 'main_wing'		
    wing.aspect_ratio	          = 10 
    wing.sweeps.quarter_chord	  = 0.
    wing.thickness_to_chord	  = 0.12 
    wing.taper	                  = 1. 
    wing.span_efficiency	  = 0.9 
    wing.spans.projected	  = 30.  * Units.feet
    wing.chords.root	          = 3.  * Units.feet
    wing.total_length	          = 3.   * Units.feet	
    wing.chords.tip	          = 3.   * Units.feet	
    wing.chords.mean_aerodynamic  = 3.   * Units.feet	
    wing.dihedral	          = 0.   * Units.degrees
    wing.areas.reference	  = 90.0 * Units.feet**2
    wing.areas.wetted	          = 180. * Units.feet**2
    wing.areas.exposed	          = 180. * Units.feet**2 
    wing.twists.root	          = 0.0  * Units.degrees
    wing.twists.tip	          = 0.0  * Units.degrees	
    wing.origin	                  = [13  *0.3048  ,  0. , 6.25*0.3048  ]
    wing.aerodynamic_center	  = [14. *0.3048 ,  0.  , 6.25*0.3048  ]
    wing.symmetric                = True
 
    # add to vehicle
    vehicle.append_component(wing)   
     
    # Tandem
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag	                  = 'tandem_wing'		
    wing.aspect_ratio	          = 10.9  
    wing.sweeps.quarter_chord	  = 0. 
    wing.thickness_to_chord	  = 0.12  
    wing.taper	                  = 1.  
    wing.span_efficiency	  = 0.9  
    wing.spans.projected	  = 30.  * Units.feet 
    wing.chords.root	          = 2.5  * Units.feet 
    wing.total_length	          = 2.5  * Units.feet	 
    wing.chords.tip	          = 2.5  * Units.feet	 
    wing.chords.mean_aerodynamic  = 2.5  * Units.feet	 
    wing.dihedral	          = 0.   * Units.degrees 	
    wing.areas.reference	  = 75.0 * Units.feet**2 
    wing.areas.wetted	          = 150. * Units.feet**2 	
    wing.areas.exposed	          = 150. * Units.feet**2 	
    wing.twists.root	          = 0.0  * Units.degrees 	
    wing.twists.tip	          = 0.0  * Units.degrees 		
    wing.origin	                  = [0.0*0.3048  ,  0. , 0.]  
    wing.aerodynamic_center	  = [0.75*0.3048 ,  0. , 0.]  
    wing.symmetric                = True
    
    # add to vehicle
    vehicle.append_component(wing)   
 
    
    # ------------------------------------------------------				
    # FUSELAGE				
    # ------------------------------------------------------				
    # FUSELAGE PROPERTIES
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                                 = 'fuselage'
    fuselage.configuration	                 = 'Tube_Wing'		
    fuselage.origin	                         = [[0. , 0.,  0.]]	
    fuselage.seats_abreast	                 = 2.		
    fuselage.seat_pitch  	                 = 3.		
    fuselage.fineness.nose	                 = 0.88 		
    fuselage.fineness.tail	                 = 1.13 		
    fuselage.lengths.nose	                 = 3.2   * Units.feet	
    fuselage.lengths.tail	                 = 6.4   * Units.feet
    fuselage.lengths.cabin	                 = 6.4 	 * Units.feet	
    fuselage.lengths.total	                 = 16.0  * Units.feet	
    fuselage.width	                         = 5.85  * Units.feet	
    fuselage.heights.maximum	                 = 4.65  * Units.feet		
    fuselage.heights.at_quarter_length	         = 3.75  * Units.feet 	
    fuselage.heights.at_wing_root_quarter_chord	 = 4.65  * Units.feet	
    fuselage.heights.at_three_quarters_length	 = 4.26  * Units.feet	
    fuselage.areas.wetted	                 = 236.  * Units.feet**2	
    fuselage.areas.front_projected	         = 0.14  * Units.feet**2	  	
    fuselage.effective_diameter 	         = 5.85  * Units.feet 	
    fuselage.differential_pressure	         = 0.	
    
    # Segment 	
    segment = SUAVE.Components.Fuselages.Segment() 
    segment.tag			                 = 'segment_1'		
    segment.origin	                         = [0., 0. ,0.]		
    segment.percent_x_location	                 = 0.		
    segment.percent_z_location	                 = 0. 	
    segment.height		                 = 0.1   * Units.feet 		
    segment.width		                 = 0.1	* Units.feet 	 		
    segment.length		                 = 0.		
    segment.effective_diameter	                 = 0.1	* Units.feet 		
    fuselage.Segments.append(segment)            
                          
    # Segment 
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                 = 'segment_2'		
    segment.origin		                 = [4.*0.3048 , 0. ,2.1*0.3048 ]  	
    segment.percent_x_location	                 = 0.25 	
    segment.percent_z_location	                 = 0.09609 
    segment.height		                 = 3.75  * Units.feet 
    segment.width		                 = 5.65  * Units.feet 	
    segment.length		                 = 3.2   * Units.feet 	
    segment.effective_diameter	                 = 5.65 	* Units.feet 
    fuselage.Segments.append(segment)  
                          
    # Segment 
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                 =' segment_3'		
    segment.origin		                 = [8.*0.3048 , 0. ,2.34*0.3048 ]  	
    segment.percent_x_location	                 = 0.5 	
    segment.percent_z_location	                 = 0.17713 
    segment.height		                 = 4.921  * Units.feet	 
    segment.width		                 = 5.90  * Units.feet 	 
    segment.length		                 = 3.2   * Units.feet
    segment.effective_diameter	                 = 5.55  * Units.feet 
    fuselage.Segments.append(segment)            
                          
    # Segment 	
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                 = 'segment_4'		
    segment.origin		                 = [12.*0.3048 , 0. ,2.77*0.3048 ] 	
    segment.percent_x_location	                 = 0.75 
    segment.percent_z_location	                 = 0.249		
    segment.height		                 = 4.73  * Units.feet		
    segment.width		                 = 4.26  * Units.feet 		
    segment.length		                 = 3.2   * Units.feet 	
    segment.effective_diameter	                 = 4.26  * Units.feet 
    fuselage.Segments.append(segment)  
                          
    # Segment
    segment = SUAVE.Components.Fuselages.Segment()
    segment.tag			                 = 'segment_5'		
    segment.origin		                 = [16.*0.3048 , 0. ,4.66*0.3048 ]   	
    segment.percent_x_location	                 = 1.0
    segment.percent_z_location	                 = 0.38 
    segment.height		                 = 0.67	* Units.feet
    segment.width		                 = 0.33	* Units.feet
    segment.length		                 = 3.2   * Units.feet	
    segment.effective_diameter	                 = 0.33  * Units.feet
    fuselage.Segments.append(segment)  

    fuselage.OpenVSP_values                      = Data()  
    fuselage.OpenVSP_values.nose                 = Data()
    fuselage.OpenVSP_values.nose.top             = Data()
    fuselage.OpenVSP_values.nose.side            = Data()
    fuselage.OpenVSP_values.nose.top.angle       = 75.00
    fuselage.OpenVSP_values.nose.top.strength    = 0.40
    fuselage.OpenVSP_values.nose.side.angle      = 45.00
    fuselage.OpenVSP_values.nose.side.strength   = 0.75 
    fuselage.OpenVSP_values.nose.TB_Sym          = True
    fuselage.OpenVSP_values.nose.z_pos           = 0.0
    
    fuselage.OpenVSP_values.tail                 = Data()
    fuselage.OpenVSP_values.tail.top             = Data()
    fuselage.OpenVSP_values.tail.side            = Data()    
    fuselage.OpenVSP_values.tail.bottom          = Data()
    fuselage.OpenVSP_values.tail.top.angle       = -90.00  
    fuselage.OpenVSP_values.tail.top.strength    = 0.10  
    fuselage.OpenVSP_values.tail.side.angle      = -90.00 
    fuselage.OpenVSP_values.tail.side.strength   = 0.10 
    fuselage.OpenVSP_values.tail.TB_Sym          = True
    fuselage.OpenVSP_values.tail.bottom.angle    = -90.00 
    fuselage.OpenVSP_values.tail.bottom.strength =  0.10 
    fuselage.OpenVSP_values.tail.z_pos           = 0.38 

    # add to vehicle
    vehicle.append_component(fuselage)    
   
    #------------------------------------------------------------------
    # PROPULSOR
    #------------------------------------------------------------------

    net = Vectored_Thrust()    
    net.number_of_engines        = 8
    net.thrust_angle             = 90.0 * Units.degrees #  conversion to radians,
    net.areas                    = Data()
    net.areas.wetted             = .05
    net.nacelle_diameter         = 0.0001
    net.engine_length            = 0.001
    net.voltage                  = 400.
                                 
    #------------------------------------------------------------------
    # Design Electronic Speed Controller 
    #------------------------------------------------------------------
    esc_lift                     = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc_lift.efficiency          = 0.95
    net.esc_lift                 = esc_lift

    esc_thrust                   = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc_thrust.efficiency        = 0.95
    net.esc_forward              = esc_thrust

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
    bat.max_voltage              = 400.
    bat.mass_properties.mass     = 300. * Units.kg
    initialize_from_mass(bat, bat.mass_properties.mass)
    net.battery                  = bat
    
    # ----------------------------------------------------------
    # PROPULSOR   
    # ----------------------------------------------------------
    prop                         = SUAVE.Components.Energy.Converters.Propeller()
    prop.tag                     = 'Vectored_Thrust_Propeller'    
    prop.y_pitch                 = 1.850
    prop.tip_radius              = 0.8875  
    prop.hub_radius              = 0.1 
    prop.number_blades           = 3   
    vehicle_weight               = vehicle.mass_properties.takeoff*9.81*0.453592    
    rho                          = 1.2
    prop.disc_area               = np.pi*(prop.tip_radius**2)    
    prop.induced_hover_velocity  = np.sqrt(vehicle_weight/(2*rho*prop.disc_area*net.number_of_engines)) 
    
    prop.freestream_velocity     = 55    * Units['mph']        
    prop.angular_velocity        = 1500. * Units['rpm']       
    prop.design_Cl               = 0.7
    prop.design_altitude         = 500   * Units.feet 
    prop.design_thrust           = (vehicle.mass_properties.takeoff/net.number_of_engines)*0.453592*9.81*1.2     
    prop.rotation                = [1,1,1,1,-1,-1,-1,-1]
    prop.x_pitch_count           = 1
    prop.y_pitch_count           = 2     
    prop.y_pitch                 = 7.5   * Units.feet 
    
    prop_origin                  = [[0*0.3048  , 7.5*0.3048  , 0*0.3048 ]]  
    prop_origin.append([0*0.3048 , 7.5*0.3048 , 0*0.3048])
    # populating propellers on one side of wing
    if prop.y_pitch_count > 1 :
        for n in range(prop.y_pitch_count):
            if n == 0:
                continue
            for i in range(len(prop_origin)):
                proppeller_origin = np.array([prop_origin[i][0] , prop_origin[i][1] +  n*prop.y_pitch ,prop_origin[i][2]])
                prop_origin.append(proppeller_origin)   
   
    # populating propellers on the other side of the vehicle  
    for n in range(len(prop_origin)):
        proppeller_origin = [prop_origin[n][0] , -prop_origin[n][1] ,prop_origin[n][2] ]
        prop_origin.append(proppeller_origin) 
            
    prop.origin = prop_origin
    # append propellers to vehicle     
    net.propeller = prop
    
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    etam                              = 0.95
    v                                 = bat.max_voltage *3/4
    omeg                              = 2500. * Units.rpm
    kv                                = 8.5   * Units.rpm
    io                                = 2.0                                      
    res                               = ((v-omeg/kv)*(1.-etam*v*kv/omeg))/io
    
    motor                             = SUAVE.Components.Energy.Converters.Motor()
    motor.mass_properties.mass        = 3. * Units.kg
    motor.origin                      = prop.origin    
    motor.speed_constant              = kv
    motor.resistance                  = res
    motor.no_load_current             = io    
    motor.gear_ratio                  = 1.0
    motor.gearbox_efficiency          = 1.0
    motor.propeller_radius            = prop.tip_radius
    net.motor                         = motor 
 
    vehicle.append_component(net)
    
    #----------------------------------------------------------------------------------------
    # Add extra drag sources from motors, props, and landing gear. All of these hand measured
    #----------------------------------------------------------------------------------------
    motor_height                            = .25                   * Units.feet
    motor_width                             =  1.6                  * Units.feet    
    propeller_width                         = 1.                    * Units.inches
    propeller_height                        = propeller_width *.12    
    main_gear_width                         = 1.5                   * Units.inches
    main_gear_length                        = 2.5                   * Units.feet    
    nose_gear_width                         = 2.                    * Units.inches
    nose_gear_length                        = 2.                    * Units.feet    
    nose_tire_height                        = (0.7 + 0.4)           * Units.feet
    nose_tire_width                         = 0.4                   * Units.feet    
    main_tire_height                        = (0.75 + 0.5)          * Units.feet
    main_tire_width                         = 4.                    * Units.inches    
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
    base_segment.state.numerics.number_control_points        = 4
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals
    base_segment.state.unknowns.propeller_power_coefficient  = 0.05 * ones_row(1) 
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)    
    
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
    
    #segment.state.unknowns.throttle                         = 0.5 * ones_row(1)
    #segment.state.unknowns.propeller_power_coefficient      = 0.02 * ones_row(1) 
    
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
    #  First Transition Segment
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "Transition_1"

    segment.analyses.extend( analyses.transition_seg_1_4 )

    segment.altitude        = 40.  * Units.ft
    segment.air_speed_start = 5. * Units['mph']       #(~27kts )
    segment.air_speed_end   = 35 * Units['mph']      #(~75 kts, 38 m/s )
    segment.acceleration    = 9.81/5
    segment.pitch_initial   = 0.0
    segment.pitch_final     = 7. * Units.degrees
           
    segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.90 * ones_row(1)  

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)
  
    # ------------------------------------------------------------------
    #  Second Transition Segment
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "Transition_2"

    segment.analyses.extend( analyses.transition_seg_2_3)

    segment.altitude        = 40.  * Units.ft
    segment.air_speed_start = 35.  * Units['mph'] 
    segment.air_speed_end   = 85.  * Units['mph'] 
    segment.acceleration    = 9.81/5
    segment.pitch_initial   = 0.0
    segment.pitch_final     = 7. * Units.degrees
           
    segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.90 * ones_row(1) 

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)
    
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
    
    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.70 * ones_row(1)
    
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
    segment.distance  = 50.    * Units.miles                       
    
    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.70 * ones_row(1)
    
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
               
    
    #segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    #segment.state.unknowns.throttle                    = 0.70 * ones_row(1)
    
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.conditions.stability    = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability = SUAVE.Methods.skip      
        
    
    # add to misison
    mission.append_segment(segment)     
    
  
    # ------------------------------------------------------------------
    #  Third Transition Segment
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "Transition_3"

    segment.analyses.extend( analyses.transition_seg_2_3)

    segment.altitude        = 40.  * Units.ft
    segment.air_speed_start = 85.  * Units['mph'] 
    segment.air_speed_end   = 35.  * Units['mph'] 
    segment.acceleration    = -9.81/5
    segment.pitch_initial   = 0.0
    segment.pitch_final     = 7. * Units.degrees
           
    #segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    #segment.state.unknowns.throttle                    = 0.90 * ones_row(1) 

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    
    # ------------------------------------------------------------------
    #  Forth Transition Segment
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Acceleration_Constant_Altitude(base_segment)
    segment.tag = "Transition_4"

    segment.analyses.extend( analyses.transition_seg_1_4)

    segment.altitude        = 40.  * Units.ft
    segment.air_speed_start = 35. * Units['mph']       #(~27kts )
    segment.air_speed_end   = 5 * Units['mph']      #(~75 kts, 38 m/s )
    segment.acceleration    = -9.81/5
    segment.pitch_initial   = 0.0
    segment.pitch_final     = 7. * Units.degrees
           
    #segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    #segment.state.unknowns.throttle                    = 0.90 * ones_row(1)  

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals    
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

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

def missions_setup(base_mission,vehicle):

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
    config.propulsors.propulsor.pitch_command = -40.  * Units.degrees    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Hover Climb Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'hover_climb'
    config.propulsors.propulsor.thrust_angle  = 90.0 * Units.degrees
    config.propulsors.propulsor.pitch_command = -40.  * Units.degrees    
    configs.append(config)

    # ------------------------------------------------------------------
    #   Hover-to-Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'transition_seg_1_4'
    config.propulsors.propulsor.thrust_angle   = 85.0  * Units.degrees
    config.propulsors.propulsor.pitch_command  = 0. * Units.degrees  
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Hover-to-Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'transition_seg_2_3'
    config.propulsors.propulsor.thrust_angle  = 75.0  * Units.degrees  
    config.propulsors.propulsor.pitch_command = 0. * Units.degrees  
    configs.append(config)
        
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    config.propulsors.propulsor.thrust_angle     =  0. * Units.degrees
    config.propulsors.propulsor.pitch_command    = 0  * Units.degrees  
    configs.append(config)  
    
    # ------------------------------------------------------------------
    #   Hover Configuration
    # ------------------------------------------------------------------
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'hover_descent'
    config.propulsors.propulsor.thrust_angle  = 89.0 * Units.degrees
    config.propulsors.propulsor.pitch_command = -20.  * Units.degrees    
    configs.append(config)
    
    return configs


# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results,vec_configs):

           
    return     
        

if __name__ == '__main__': 
    main()    
    plt.show()
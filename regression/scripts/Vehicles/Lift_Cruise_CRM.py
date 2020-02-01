# Lift_Cruise_CRM.py
# 
# Created: May 2019, M Clarke

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data 
import copy
from SUAVE.Components.Energy.Networks.Lift_Cruise import Lift_Cruise
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass
from SUAVE.Methods.Propulsion import propeller_design   
from SUAVE.Methods.Weights.Buildups.Electric_Lift_Cruise.empty import empty

import numpy as np
import pylab as plt
from copy import deepcopy 

# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # build the vehicle, configs, and analyses
    configs, analyses = full_setup()
    
    # configs.finalize()
    analyses.finalize()    
    
    # weight analysis
    weights = analyses.weights
    breakdown = weights.evaluate()          
    
    # mission analysis
    mission = analyses.mission
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

    # vehicle analyses
    analyses = base_analysis(vehicle)

    # mission analyses
    mission  = mission_setup(analyses,vehicle)

    analyses.mission = mission
    
    return  vehicle, analyses

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    vehicle               = SUAVE.Vehicle()
    vehicle.tag           = 'Lift_Cruise_CRM'
    vehicle.configuration = 'eVTOL'
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff           = 2450. * Units.lb 
    vehicle.mass_properties.operating_empty   = 2250. * Units.lb               # Approximate
    vehicle.mass_properties.max_takeoff       = 2450. * Units.lb               # Approximate
    vehicle.mass_properties.center_of_gravity = [2.0144,   0.  ,  0. ] # Approximate
    
    # basic parameters 
    vehicle.reference_area                    = 10.76 	
    vehicle.envelope.ultimate_load            = 5.7   
    vehicle.envelope.limit_load               = 3.  
    
    # ------------------------------------------------------				
    # WINGS				
    # ------------------------------------------------------				
    # WING PROPERTIES	
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag	                  = 'main_wing'		
    wing.aspect_ratio	          = 10.76 
    wing.sweeps.quarter_chord	  = 0.0	 * Units.degrees
    wing.thickness_to_chord	  = 0.18		
    wing.taper	                  = 1. 	
    wing.span_efficiency	  = 0.9		
    wing.spans.projected	  = 35.0   * Units.feet
    wing.chords.root	          = 3.25   * Units.feet
    wing.total_length	          = 3.25   * Units.feet	
    wing.chords.tip	          = 3.25   * Units.feet	
    wing.chords.mean_aerodynamic  = 3.25   * Units.feet		
    wing.dihedral	          = 1.0    * Units.degrees		
    wing.areas.reference	  = 113.75 * Units.feet**2	
    wing.areas.wetted	          = 227.5  * Units.feet**2		
    wing.areas.exposed	          = 227.5  * Units.feet**2		
    wing.twists.root	          = 4.0    * Units.degrees		
    wing.twists.tip	          = 0.0    * Units.degrees			
    wing.origin	                  = [1.5, 0., 0. ] 
    wing.aerodynamic_center	  = [1.975 , 0., 0.]    
    wing.winglet_fraction         = 0.0  
    wing.symmetric                = True
    wing.vertical                 = False
    
    # Segment 	
    segment = SUAVE.Components.Wings.Segment()
    segment.tag			  = 'Section_1'			
    segment.percent_span_location = 0.		
    segment.twist		  = 0.		
    segment.root_chord_percent	  = 1.5	
    segment.dihedral_outboard	  = 1.0     * Units.degrees
    segment.sweeps.quarter_chord  = 8.5     * Units.degrees
    segment.thickness_to_chord	  = 0.18		
    wing.Segments.append(segment)
    
    # Segment 
    segment = SUAVE.Components.Wings.Segment()
    segment.tag			  = 'Section_2'				
    segment.percent_span_location = 0.227	
    segment.twist		  = 0.		
    segment.root_chord_percent	  = 1. 	
    segment.dihedral_outboard	  = 1.0  * Units.degrees
    segment.sweeps.quarter_chord  = 0.0	 * Units.degrees	
    segment.thickness_to_chord	  = 0.12	
    wing.Segments.append(segment)
                          
    # Segment 
    segment = SUAVE.Components.Wings.Segment()
    segment.tag			  = 'Section_3'			
    segment.percent_span_location = 1.0 
    segment.twist		  = 0.		
    segment.root_chord_percent	  = 1.0 
    segment.dihedral_outboard	  = 1.0  * Units.degrees
    segment.sweeps.quarter_chord  = 0.0 * Units.degrees
    segment.thickness_to_chord	  = 0.12	
    wing.Segments.append(segment) 
       
    # add to vehicle
    vehicle.append_component(wing)       
    
    # WING PROPERTIES
    wing = SUAVE.Components.Wings.Wing()
    wing.tag		         = 'horizontal_tail'		
    wing.aspect_ratio		 = 4.0	
    wing.sweeps.quarter_chord	 = 0.0		
    wing.thickness_to_chord	 = 0.12		
    wing.taper			 = 1.0		
    wing.span_efficiency	 = 0.9		
    wing.spans.projected	 = 8.0	 * Units.feet
    wing.chords.root		 = 2.0	 * Units.feet	
    wing.total_length		 = 2.0	 * Units.feet	
    wing.chords.tip		 = 2.0	 * Units.feet	
    wing.chords.mean_aerodynamic = 2.0	 * Units.feet			
    wing.dihedral		 = 0.	 * Units.degrees		
    wing.areas.reference	 = 16.0  * Units.feet**2	
    wing.areas.wetted		 = 32.0  * Units.feet**2	 		
    wing.areas.exposed		 = 32.0  * Units.feet**2	 
    wing.twists.root		 = 0.	 * Units.degrees		
    wing.twists.tip		 = 0.	 * Units.degrees		
    wing.origin		         = [14.0*0.3048 , 0.0 , 0.205 ] 		
    wing.aerodynamic_center	 = [15.0*0.3048 ,  0.,  0.] 
    wing.symmetric               = True    
    
    # add to vehicle
    vehicle.append_component(wing)    
    
    
    # WING PROPERTIES
    wing = SUAVE.Components.Wings.Wing()
    wing.tag		         = 'vertical_tail_1'
    wing.aspect_ratio		 = 2.	
    wing.sweeps.quarter_chord	 = 20.0 * Units.degrees	
    wing.thickness_to_chord	 = 0.12
    wing.taper			 = 0.5
    wing.span_efficiency	 = 0.9 
    wing.spans.projected	 = 3.0	* Units.feet	
    wing.chords.root		 = 2.0	* Units.feet		
    wing.total_length		 = 2.0	* Units.feet	
    wing.chords.tip		 = 1.0	* Units.feet		
    wing.chords.mean_aerodynamic = 1.5	* Units.feet
    wing.areas.reference	 = 4.5 	* Units.feet**2
    wing.areas.wetted		 = 9.0 	* Units.feet**2	
    wing.areas.exposed		 = 9.0 	* Units.feet**2	
    wing.twists.root		 = 0.	* Units.degrees	
    wing.twists.tip		 = 0.	* Units.degrees		
    wing.origin		         = [14.0*0.3048 , 4.0*0.3048  , 0.205  ] 		
    wing.aerodynamic_center	 = 0.0 		
    wing.winglet_fraction        = 0.0  
    wing.vertical		 = True	
    wing.symmetric               = False
    
    # add to vehicle
    vehicle.append_component(wing)   
    
    
    # WING PROPERTIES
    wing = SUAVE.Components.Wings.Wing()
    wing.tag		         = 'vertical_tail_2'
    wing.aspect_ratio		 = 2.	
    wing.sweeps.quarter_chord	 = 20.0 * Units.degrees	
    wing.thickness_to_chord	 = 0.12
    wing.taper			 = 0.5
    wing.span_efficiency	 = 0.9 
    wing.spans.projected	 = 3.0	* Units.feet	
    wing.chords.root		 = 2.0	* Units.feet		
    wing.total_length		 = 2.0	* Units.feet	
    wing.chords.tip		 = 1.0	* Units.feet		
    wing.chords.mean_aerodynamic = 1.5	* Units.feet
    wing.areas.reference	 = 4.5 	* Units.feet**2
    wing.areas.wetted		 = 9.0 	* Units.feet**2	
    wing.areas.exposed		 = 9.0 	* Units.feet**2	
    wing.twists.root		 = 0.	* Units.degrees		
    wing.twists.tip		 = 0.	* Units.degrees			
    wing.origin		         = [14.0*0.3048 , -4.0*0.3048  , 0.205   ] 	
    wing.aerodynamic_center	 = 0.0 		
    wing.winglet_fraction        = 0.0  
    wing.vertical		 = True	  
    wing.symmetric               = False
    
    # add to vehicle
    vehicle.append_component(wing)   
    
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
    
    # add to vehicle
    vehicle.append_component(fuselage)    
    
    #-------------------------------------------------------------------
    # BOOMS			
    #-------------------------------------------------------------------   
    boom = SUAVE.Components.Fuselages.Fuselage()
    boom.tag                                    = 'Boom_1R'
    boom.configuration	                        = 'Boom'		
    boom.origin	                                = [[0.718,7.5*0.3048 , -0.15 ]]
    boom.seats_abreast	                        = 0.		
    boom.seat_pitch	                        = 0.0	
    boom.fineness.nose	                        = 0.950 		
    boom.fineness.tail	                        = 1.029 		
    boom.lengths.nose	                        = 0.5   * Units.feet			
    boom.lengths.tail	                        = 0.5   * Units.feet	 		
    boom.lengths.cabin	                        = 9.	* Units.feet
    boom.lengths.total	                        = 10    * Units.feet			
    boom.width	                                = 0.5	* Units.feet			
    boom.heights.maximum                        = 0.5	* Units.feet			
    boom.heights.at_quarter_length	        = 0.5	* Units.feet			
    boom.heights.at_three_quarters_length	= 0.5	* Units.feet
    boom.heights.at_wing_root_quarter_chord     = 0.5   * Units.feet
    boom.areas.wetted		                = 18 	* Units.feet**2
    boom.areas.front_projected	                = 0.26 	* Units.feet**2
    boom.effective_diameter	                = 0.5	* Units.feet	 		
    boom.differential_pressure	                = 0.	
    boom.y_pitch_count                          = 2
    boom.y_pitch                                = (72/12)*0.3048
    boom.symmetric                              = True
    boom.boom_pitch                             = 6 * Units.feet
    boom.index = 1
    
    # add to vehicle
    vehicle.append_component(boom)    
    
    # create pattern of booms on one side
    original_boom_origin =  boom.origin	
    if boom.y_pitch_count >  1 : 
        for n in range(boom.y_pitch_count):
            if n == 0:
                continue
            else:
                index = n+1
                boom = deepcopy(vehicle.fuselages.boom_1r)
                boom.origin[0][1] = n*boom.boom_pitch + original_boom_origin[0][1]
                boom.tag = 'Boom_' + str(index) + 'R'
                boom.index = n 
                vehicle.append_component(boom)
    
    if boom.symmetric : 
        for n in range(boom.y_pitch_count):
            index = n+1
            boom = deepcopy(vehicle.fuselages.boom_1r)
            boom.origin[0][1] = -n*boom.boom_pitch - original_boom_origin[0][1]
            boom.tag = 'Boom_' + str(index) + 'L'
            boom.index = n 
            vehicle.append_component(boom) 
            
            
    #------------------------------------------------------------------
    # PROPULSOR
    #------------------------------------------------------------------
    net = Lift_Cruise()
    net.number_of_engines_lift    = 12
    net.number_of_engines_forward = 1
    net.thrust_angle_lift         = 90. * Units.degrees
    net.thrust_angle_forward      = 0. 
    net.nacelle_diameter          = 0.6 * Units.feet	# need to check	
    net.engine_length             = 0.5 * Units.feet
    net.areas                     = Data()
    net.areas.wetted              = np.pi*net.nacelle_diameter*net.engine_length + 0.5*np.pi*net.nacelle_diameter**2    
    net.voltage                   = 500.

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
    bat.max_voltage              = net.voltage 
    bat.mass_properties.mass     = 300. * Units.kg
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
    
    # Thrust Propeller
    prop_forward                     = SUAVE.Components.Energy.Converters.Propeller()
    prop_forward.tag                 = 'Forward_Prop'
    prop_forward.number_blades       = 3
    prop_forward.number_of_engines   = net.number_of_engines_forward
    prop_forward.freestream_velocity = 110.   * Units['mph']
    prop_forward.tip_radius          = 1.0668
    prop_forward.hub_radius          = 0.21336 
    prop_forward.design_tip_mach     = 0.65
    prop_forward.angular_velocity    = (prop_forward.design_tip_mach *speed_of_sound*rad_per_sec_to_rpm /prop_forward.tip_radius)  * Units['rpm']  
    prop_forward.design_Cl           = 0.7
    prop_forward.design_altitude     = 1. * Units.km 
    Drag                             = vehicle.reference_area * (0.5*rho*prop_forward.freestream_velocity**2 )*Cd
    prop_forward.design_thrust       = (Drag*2.5)/net.number_of_engines_forward
    prop_forward.design_power        = 0. * Units.watts 
    prop_forward                     = propeller_design(prop_forward)   
    prop_forward.origin              = [[16.*0.3048 , 0. ,2.02*0.3048 ]]  
    net.propeller_forward            = prop_forward
    
    # Lift Rotors
    prop_lift                        = SUAVE.Components.Energy.Converters.Propeller()
    prop_lift.tag                    = 'Lift_Prop'
    prop_lift.tip_radius             = 2.8 * Units.feet
    prop_lift.hub_radius             = 0.35 * Units.feet      
    prop_lift.number_blades          = 2   
    prop_lift.design_tip_mach        = 0.65
    prop_lift.number_of_engines      = net.number_of_engines_lift
    prop_lift.disc_area              = np.pi*(prop_lift.tip_radius**2)     
    Lift                             = vehicle.mass_properties.takeoff*9.81     
    prop_lift.induced_hover_velocity = np.sqrt(Lift/(2*rho*prop_lift.disc_area*net.number_of_engines_lift)) 
    prop_lift.freestream_velocity    = 500. * Units['ft/min'] # hover and climb rate  
    prop_lift.angular_velocity       = prop_lift.design_tip_mach* speed_of_sound* rad_per_sec_to_rpm /prop_lift.tip_radius  * Units['rpm']      
    prop_lift.design_Cl              = 0.7
    prop_lift.design_altitude        = 20 * Units.feet                            
    prop_lift.design_thrust          = (Lift * 2.5 )/net.number_of_engines_lift 
    prop_lift.design_power           = 0.0  
    prop_lift.x_pitch_count          = 2 
    prop_lift.y_pitch_count          = vehicle.fuselages['boom_1r'].y_pitch_count
    prop_lift.y_pitch                = vehicle.fuselages['boom_1r'].y_pitch 
    prop_lift                        = propeller_design(prop_lift)          
    prop_lift.origin                 = vehicle.fuselages['boom_1r'].origin
    prop_lift.symmetric              = True
    
    # populating propellers on one side of wing
    if prop_lift.y_pitch_count > 1 :
        for n in range(prop_lift.y_pitch_count):
            if n == 0:
                continue
            proppeller_origin = [prop_lift.origin[0][0] , prop_lift.origin[0][1] +  n*prop_lift.y_pitch ,prop_lift.origin[0][2]]
            prop_lift.origin.append(proppeller_origin)   
   
   
    # populating propellers on one side of the vehicle 
    if prop_lift.x_pitch_count > 1 :
        relative_prop_origins = np.linspace(0,vehicle.fuselages['boom_1r'].lengths.total,prop_lift.x_pitch_count)
        for n in range(len(prop_lift.origin)):
            for m in range(len(relative_prop_origins)-1):
                proppeller_origin = [prop_lift.origin[n][0] + relative_prop_origins[m+1] , prop_lift.origin[n][1] ,prop_lift.origin[n][2] ]
                prop_lift.origin.append(proppeller_origin)
                 
    # propulating propellers on the other side of thevehicle   
    if prop_lift.symmetric : 
        for n in range(len(prop_lift.origin)):
            proppeller_origin = [prop_lift.origin[n][0] , -prop_lift.origin[n][1] ,prop_lift.origin[n][2] ]
            prop_lift.origin.append(proppeller_origin) 
    
    # re-compute number of lift propellers if changed 
    net.number_of_engines_lift    = len(prop_lift.origin)        
    
    # append propellers to vehicle     
    net.propeller_lift = prop_lift
    
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    # Propeller (Thrust) motor
    motor_forward                      = SUAVE.Components.Energy.Converters.Motor()
    etam                               = 0.95
    v                                  = bat.max_voltage *3/4
    omeg                               = prop_forward.angular_velocity  
    io                                 = 2.0 
    start_kv                           = 1
    end_kv                             = 15 
    # do optimization to find kv or just do a linspace then remove all negative values, take smallest one use 0.05 change
    # essentially you are sizing the motor for a particular rpm which is sized for a design tip mach 
    # this reduces the bookkeeping errors     
    possible_kv_vals                   = np.linspace(start_kv,end_kv,(end_kv-start_kv)*20 +1 , endpoint = True) * Units.rpm
    res_kv_vals                        = ((v-omeg/possible_kv_vals)*(1.-etam*v*possible_kv_vals/omeg))/io  
    positive_res_vals                  = np.extract(res_kv_vals > 0 ,res_kv_vals) 
    kv_idx                             = np.where(res_kv_vals == min(positive_res_vals))[0][0]   
    kv                                 = possible_kv_vals[kv_idx]  
    res                                = min(positive_res_vals) 
    
    motor_forward.mass_properties.mass = 2.0  * Units.kg
    motor_forward.origin               = prop_forward.origin  
    motor_forward.propeller_radius     = prop_forward.tip_radius   
    motor_forward.speed_constant       = kv
    motor_forward.resistance           = res
    motor_forward.no_load_current      = io 
    motor_forward.gear_ratio           = 1. 
    motor_forward.gearbox_efficiency   = 1. # Gear box efficiency     
    net.motor_forward                  = motor_forward
    
    # Rotor (Lift) Motor
    motor_lift                         = SUAVE.Components.Energy.Converters.Motor()
    etam                               = 0.95
    v                                  = bat.max_voltage     
    omeg                               = prop_lift.angular_velocity  
    io                                 = 4.0
    start_kv                           = 1
    end_kv                             = 15 
    # do optimization to find kv or just do a linspace then remove all negative values, take smallest one use 0.05 change
    # essentially you are sizing the motor for a particular rpm which is sized for a design tip mach 
    # this reduces the bookkeeping errors     
    possible_kv_vals                   = np.linspace(start_kv,end_kv,(end_kv-start_kv)*20 +1 , endpoint = True) * Units.rpm
    res_kv_vals                        = ((v-omeg/possible_kv_vals)*(1.-etam*v*possible_kv_vals/omeg))/io  
    positive_res_vals                   = np.extract(res_kv_vals > 0 ,res_kv_vals) 
    kv_idx                             = np.where(res_kv_vals == min(positive_res_vals))[0][0]   
    kv                                 = possible_kv_vals[kv_idx]  
    res                                = min(positive_res_vals)
    
    motor_lift.mass_properties.mass    = 3. * Units.kg 
    motor_lift.origin                  = prop_lift.origin  
    motor_lift.propeller_radius        = prop_lift.tip_radius
    motor_lift.speed_constant          = kv
    motor_lift.resistance              = res
    motor_lift.no_load_current         = io    
    motor_lift.gear_ratio              = 1.0
    motor_lift.gearbox_efficiency      = 1.0
    net.motor_lift                     = motor_lift 

    # append motor origin spanwise locations onto wing data structure 
    vehicle.append_component(net)
    
    #----------------------------------------------------------------------------------------
    # Add extra drag sources from motors, props, and landing gear. All of these hand measured
    #----------------------------------------------------------------------------------------
    motor_height                            = .25 * Units.feet
    motor_width                             =  1.6 * Units.feet    
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
    
    vehicle.wings['main_wing'].motor_spanwise_locations = np.multiply(
        2./36.25,
        [-5.435, -5.435, -9.891, -9.891, -14.157, -14.157,
         5.435, 5.435, 9.891, 9.891, 14.157, 14.157])

    vehicle.wings['main_wing'].winglet_fraction = 0.0
    vehicle.wings['main_wing'].thickness_to_chord = 0.18
    vehicle.wings['main_wing'].chords.mean_aerodynamic = 0.9644599977664836
    
    ##----------------------------------------------------------------------------------------
    ## EVALUATE WEIGHTS using calculation (battery not updated) 
    ##----------------------------------------------------------------------------------------
    #vehicle.weight_breakdown                = empty(vehicle)
    #MTOW                                    = vehicle.weight_breakdown.total
    #Payload                                 = vehicle.weight_breakdown.payload
    #OE                                      = MTOW - vehicle.weight_breakdown.battery               
    #vehicle.mass_properties.takeoff         = MTOW
    #vehicle.mass_properties.operating_empty = OE  
    
    
    return vehicle

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
    weights = SUAVE.Analyses.Weights.Weights_Electric_Lift_Cruise()
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
    mission            = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag        = 'the_mission'

    # airport
    airport            = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport    = airport    

    # unpack Segments module
    Segments                                                 = SUAVE.Analyses.Mission.Segments
                                                             
    # base segment                                           
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row
    base_segment.state.numerics.number_control_points        = 10
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_transition
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_transition
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
    #   First Taxi Segment: Constant Speed
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "Ground_Taxi"

    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses )                                                            
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 40.  * Units.ft
    segment.climb_rate                                       = 500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.propulsors.propulsor.battery.max_energy*0.95
                                                             
    segment.state.unknowns.propeller_power_coefficient_lift = 0.04 * ones_row(1)
    segment.state.unknowns.throttle_lift                    = 0.85 * ones_row(1)
    segment.state.unknowns.__delitem__('throttle')

    segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_no_forward
    segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_no_forward       
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------

    segment = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag = "transition_1"

    segment.analyses.extend( analyses )

    segment.altitude        = 40.  * Units.ft
    segment.air_speed_start = 0.   * Units['ft/min']
    segment.air_speed_end   = 1.2 * Vstall
    segment.acceleration    = 9.81/5
    segment.pitch_initial   = 0.0
    segment.pitch_final     = 5. * Units.degrees

    segment.state.unknowns.propeller_power_coefficient_lift = 0.07 * ones_row(1)
    segment.state.unknowns.throttle_lift                    = 0.70 * ones_row(1) 
    segment.state.unknowns.propeller_power_coefficient      = 0.03 * ones_row(1)
    segment.state.unknowns.throttle                         = .60  * ones_row(1)   
    segment.state.residuals.network                         = 0.   * ones_row(3)    

    segment.process.iterate.unknowns.network                = vehicle.propulsors.propulsor.unpack_unknowns_transition
    segment.process.iterate.residuals.network               = vehicle.propulsors.propulsor.residuals_transition    
    segment.process.iterate.unknowns.mission                = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability            = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability         = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"

    segment.analyses.extend( analyses )

    segment.air_speed       = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    segment.altitude_start  = 40.0 * Units.ft
    segment.altitude_end    = 300. * Units.ft
    segment.climb_rate      = 500. * Units['ft/min']

    segment.state.unknowns.propeller_power_coefficient         = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                            = 0.70 * ones_row(1)
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift     

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Cruise Segment: Constant Speed, Constant Altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag = "departure_terminal_procedures"

    segment.analyses.extend( analyses )

    segment.altitude  = 300.0 * Units.ft
    segment.time      = 60.   * Units.second
    segment.air_speed = 1.2*Vstall

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift     


    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Acceleration, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "accelerated_climb"

    segment.analyses.extend( analyses )

    segment.altitude_start  = 300.0 * Units.ft
    segment.altitude_end    = 1000. * Units.ft
    segment.climb_rate      = 500.  * Units['ft/min']
    segment.air_speed_start = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    segment.air_speed_end   = 110.  * Units['mph']                                            

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift  


    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Third Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"

    segment.analyses.extend( analyses )

    segment.altitude  = 1000.0 * Units.ft
    segment.air_speed = 110.   * Units['mph']
    segment.distance  = 60.    * Units.miles                       

    segment.state.unknowns.propeller_power_coefficient = 0.02 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.40 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift    


    # add to misison
    mission.append_segment(segment)     

    # ------------------------------------------------------------------
    #   First Descent Segment: Constant Acceleration, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "decelerating_descent"

    segment.analyses.extend( analyses )  
    segment.altitude_start  = 1000.0 * Units.ft
    segment.altitude_end    = 300. * Units.ft
    segment.climb_rate      = -500.  * Units['ft/min']
    segment.air_speed_start = 110.  * Units['mph']
    segment.air_speed_end   = 1.2*Vstall

    segment.state.unknowns.propeller_power_coefficient = 0.03 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.5 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift     

    # add to misison
    mission.append_segment(segment)        

    # ------------------------------------------------------------------
    #   Fourth Cruise Segment: Constant Speed, Constant Altitude
    # ------------------------------------------------------------------

    segment = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag = "arrival_terminal_procedures"

    segment.analyses.extend( analyses )

    segment.altitude        = 300.   * Units.ft
    segment.air_speed       = 1.2*Vstall
    segment.time            = 60 * Units.seconds

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift   

    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Second Descent Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"

    segment.analyses.extend( analyses )

    segment.altitude_start  = 300.0 * Units.ft
    segment.altitude_end    = 40. * Units.ft
    segment.climb_rate      = -400.  * Units['ft/min']  # Uber has 500->300
    segment.air_speed_start = np.sqrt((400 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    segment.air_speed_end   = 1.2*Vstall                           

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_lift 


    # add to misison
    mission.append_segment(segment)       

    # ------------------------------------------------------------------
    #   Fifth Cuise Segment: Transition
    # ------------------------------------------------------------------ 
    segment = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag = "transition_2"

    segment.analyses.extend( analyses )

    segment.altitude        = 40. * Units.ft
    segment.air_speed_start = 1.2 * Vstall      
    segment.air_speed_end   = 0 
    segment.acceleration    = -9.81/20
    segment.pitch_initial   = 5. * Units.degrees   
    segment.pitch_final     = 10. * Units.degrees      
    

    segment.state.unknowns.propeller_power_coefficient_lift = 0.04 * ones_row(1)
    segment.state.unknowns.throttle_lift                    = 0.60 * ones_row(1) 
    segment.state.unknowns.propeller_power_coefficient      = 0.05 * ones_row(1)
    segment.state.unknowns.throttle                         = .40  * ones_row(1)   
    segment.state.residuals.network                         = 0.   * ones_row(3)   
    
    
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_transition
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_transition    
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip 
    # add to misison
    mission.append_segment(segment)

      
    # ------------------------------------------------------------------
    #   Third Descent Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------

    segment = Segments.Hover.Descent(base_segment)
    segment.tag = "descent_1"

    segment.analyses.extend( analyses )

    segment.altitude_start  = 40.0  * Units.ft
    segment.altitude_end    = 0.   * Units.ft
    segment.descent_rate    = 300. * Units['ft/min']
    segment.battery_energy  = vehicle.propulsors.propulsor.battery.max_energy

    segment.state.unknowns.propeller_power_coefficient_lift = 0.03* ones_row(1)
    segment.state.unknowns.throttle_lift                    = 0.9 * ones_row(1)

    segment.state.unknowns.__delitem__('throttle')
    segment.process.iterate.unknowns.network  = vehicle.propulsors.propulsor.unpack_unknowns_no_forward
    segment.process.iterate.residuals.network = vehicle.propulsors.propulsor.residuals_no_forward    
    segment.process.iterate.unknowns.mission  = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability      = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability   = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)       

    return mission


# ----------------------------------------------------------------------
#   Plot Results
# ----------------------------------------------------------------------
def plot_mission(results,vec_configs):
    plot_electronic_conditions(results) 
    plot_lift_cruise_network(results) 
           
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
        energy   = results.segments[i].conditions.propulsion.battery_energy[:,0] 
        volts    = results.segments[i].conditions.propulsion.voltage_under_load[:,0] 
        volts_oc = results.segments[i].conditions.propulsion.voltage_open_circuit[:,0]     
        current = results.segments[i].conditions.propulsion.current[:,0]      
        battery_amp_hr = (energy*0.000277778)/volts
        C_rating   = current/battery_amp_hr
        
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, -power, line_color)
        axes.set_ylabel('Battery Power (Watts)',axis_font)
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')   
        axes.grid(True)       
    
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
#   Lift-Cruise Network
# ------------------------------------------------------------------
def plot_lift_cruise_network(results, line_color = 'bo-', save_figure = False, save_filename = "Lift_Cruise_Network"):
    axis_font = {'size':'14'} 
    # ------------------------------------------------------------------
    #   Electronic Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Lift_Cruise_Electric_Conditions")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):          
        time           = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        eta            = results.segments[i].conditions.propulsion.throttle[:,0]
        eta_l          = results.segments[i].conditions.propulsion.throttle_lift[:,0]
        energy         = results.segments[i].conditions.propulsion.battery_energy[:,0]*0.000277778
        specific_power = results.segments[i].conditions.propulsion.battery_specfic_power[:,0]
        volts          = results.segments[i].conditions.propulsion.voltage_under_load[:,0] 
        volts_oc       = results.segments[i].conditions.propulsion.voltage_open_circuit[:,0]  
                    
        axes = fig.add_subplot(2,2,1)
        axes.plot(time, eta, 'bo-',label='Forward Motor')
        axes.plot(time, eta_l, 'r^-',label='Lift Motors')
        axes.set_ylabel('Throttle')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
        plt.ylim((0,1))
        if i == 0:
            axes.legend(loc='upper center')         
    
        axes = fig.add_subplot(2,2,2)
        axes.plot(time, energy, 'bo-')
        axes.set_ylabel('Battery Energy (W-hr)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,2,3)
        axes.plot(time, volts, 'bo-',label='Under Load')
        axes.plot(time,volts_oc, 'ks--',label='Open Circuit')
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Battery Voltage (Volts)')  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)
        if i == 0:
            axes.legend(loc='upper center')                
        
        axes = fig.add_subplot(2,2,4)
        axes.plot(time, specific_power, 'bo-') 
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Specific Power')  
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
        
        
    
    if save_figure:
        plt.savefig("Lift_Cruise_Electric_Conditions.png")
    
   
    ## ------------------------------------------------------------------
    ##   Propulsion Conditions
    ## ------------------------------------------------------------------
    #fig = plt.figure("Prop-Rotor Network")
    #fig.set_size_inches(16, 8)
    #for i in range(len(results.segments)):          
        #time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        #prop_rpm    = results.segments[i].conditions.propulsion.rpm_forward [:,0] 
        #prop_thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        #prop_torque = results.segments[i].conditions.propulsion.motor_torque_forward[:,0]
        #prop_effp   = results.segments[i].conditions.propulsion.propeller_efficiency_forward[:,0]
        #prop_effm   = results.segments[i].conditions.propulsion.motor_efficiency_forward[:,0]
        #prop_Cp     = results.segments[i].conditions.propulsion.propeller_power_coefficient[:,0]
        #rotor_rpm    = results.segments[i].conditions.propulsion.rpm_lift[:,0] 
        #rotor_thrust = -results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        #rotor_torque = results.segments[i].conditions.propulsion.motor_torque_lift
        #rotor_effp   = results.segments[i].conditions.propulsion.propeller_efficiency_lift[:,0]
        #rotor_effm   = results.segments[i].conditions.propulsion.motor_efficiency_lift[:,0] 
        #rotor_Cp = results.segments[i].conditions.propulsion.propeller_power_coefficient_lift[:,0]        
    
        #axes = fig.add_subplot(2,3,1)
        #axes.plot(time, prop_rpm, 'bo-')
        #axes.plot(time, rotor_rpm, 'r^-')
        #axes.set_ylabel('RPM')
        #axes.minorticks_on()
        #axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        #axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        #axes.grid(True)       
    
        #axes = fig.add_subplot(2,3,2)
        #axes.plot(time, prop_thrust, 'bo-')
        #axes.plot(time, rotor_thrust, 'r^-')
        #axes.set_ylabel('Thrust (N)')
        #axes.minorticks_on()
        #axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        #axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        #axes.grid(True)   
    
        #axes = fig.add_subplot(2,3,3)
        #axes.plot(time, prop_torque, 'bo-' )
        #axes.plot(time, rotor_torque, 'r^-' )
        #axes.set_xlabel('Time (mins)')
        #axes.set_ylabel('Torque (N-m)')
        #axes.minorticks_on()
        #axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        #axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        #axes.grid(True)   
    
        #axes = fig.add_subplot(2,3,4)
        #axes.plot(time, prop_effp, 'bo-' )
        #axes.plot(time, rotor_effp, 'r^-' )
        #axes.set_xlabel('Time (mins)')
        #axes.set_ylabel(r'Propeller Efficiency, $\eta_{propeller}$')
        #axes.minorticks_on()
        #axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        #axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        #axes.grid(True)           
        #plt.ylim((0,1))
    
        #axes = fig.add_subplot(2,3,5)
        #axes.plot(time, prop_effm, 'bo-' )
        #axes.plot(time, rotor_effm, 'r^-' )
        #axes.set_xlabel('Time (mins)')
        #axes.set_ylabel(r'Motor Efficiency, $\eta_{motor}$')
        #axes.minorticks_on()
        #axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        #axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        #axes.grid(True)         
        #plt.ylim((0,1))
    
        #axes = fig.add_subplot(2,3,6)
        #axes.plot(time, prop_Cp, 'bo-' )
        #axes.plot(time, rotor_Cp, 'r^-'  )
        #axes.set_xlabel('Time (mins)')
        #axes.set_ylabel('Power Coefficient')
        #axes.minorticks_on()
        #axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        #axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')  
        #axes.grid(True) 
        
    #if save_figure:
        #plt.savefig("Propulsor Network.png")
            
    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Rotor")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):          
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.rpm_lift [:,0] 
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,2]
        torque = results.segments[i].conditions.propulsion.motor_torque_lift[:,0]
        effp   = results.segments[i].conditions.propulsion.propeller_efficiency_lift[:,0]
        effm   = results.segments[i].conditions.propulsion.motor_efficiency_lift[:,0] 
        Cp     = results.segments[i].conditions.propulsion.propeller_power_coefficient_lift[:,0]
    
        axes = fig.add_subplot(2,3,1)
        axes.plot(time, rpm, 'r^-')
        axes.set_ylabel('RPM')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
    
        axes = fig.add_subplot(2,3,2)
        axes.plot(time, -thrust, 'r^-')
        axes.set_ylabel('Thrust (N)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,3)
        axes.plot(time, torque, 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Torque (N-m)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,4)
        axes.plot(time, effp, 'r^-',label= r'$\eta_{rotor}$' ) 
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Propeller Efficiency $\eta_{rotor}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')   
        #if i == 0:
            #axes.legend(loc='upper center')   
        axes.grid(True)           
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,5)
        axes.plot(time, effm, 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Motor Efficiency $\eta_{mot}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        plt.ylim((0,1))
        axes.grid(True)  
    
        axes = fig.add_subplot(2,3,6)
        axes.plot(time, Cp , 'r^-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Power Coefficient')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')    
        axes.grid(True)           
    
    if save_figure:
        plt.savefig("Rotor.png")  
        
        
    # ------------------------------------------------------------------
    #   Propulsion Conditions
    # ------------------------------------------------------------------
    fig = plt.figure("Propeller")
    fig.set_size_inches(16, 8)
    for i in range(len(results.segments)):          
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        rpm    = results.segments[i].conditions.propulsion.rpm_forward [:,0] 
        thrust = results.segments[i].conditions.frames.body.thrust_force_vector[:,0]
        torque = results.segments[i].conditions.propulsion.motor_torque_forward
        effp   = results.segments[i].conditions.propulsion.propeller_efficiency_forward[:,0]
        effm   = results.segments[i].conditions.propulsion.motor_efficiency_forward[:,0]
        Cp     = results.segments[i].conditions.propulsion.propeller_power_coefficient[:,0]
    
        axes = fig.add_subplot(2,3,1)
        axes.plot(time, rpm, 'bo-')
        axes.set_ylabel('RPM')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey') 
        axes.grid(True)       
    
        axes = fig.add_subplot(2,3,2)
        axes.plot(time, thrust, 'bo-')
        axes.set_ylabel('Thrust (N)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')       
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,3)
        axes.plot(time, torque, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Torque (N-m)')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)   
    
        axes = fig.add_subplot(2,3,4)
        axes.plot(time, effp, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Propeller Efficiency $\eta_{propeller}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)           
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,5)
        axes.plot(time, effm, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel(r'Motor Efficiency $\eta_{motor}$')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')      
        axes.grid(True)         
        plt.ylim((0,1))
    
        axes = fig.add_subplot(2,3,6)
        axes.plot(time, Cp, 'bo-' )
        axes.set_xlabel('Time (mins)')
        axes.set_ylabel('Power Coefficient')
        axes.minorticks_on()
        axes.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
        axes.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')  
        axes.grid(True) 
        
    if save_figure:
        plt.savefig("Cruise_Propulsor.png") 
       
        
    return

if __name__ == '__main__': 
    main()    
    plt.show()
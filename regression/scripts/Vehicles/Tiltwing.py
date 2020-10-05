# Tiltwing.py
# 
# Created: May 2019, M Clarke
#          Sep 2020, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data
import copy
from SUAVE.Components.Energy.Networks.Vectored_Thrust import Vectored_Thrust
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_mass , size_optimal_motor
from SUAVE.Methods.Propulsion import propeller_design 
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff 
from SUAVE.Methods.Weights.Buildups.Electric_Vectored_Thrust.empty import empty
from SUAVE.Methods.Utilities.Chebyshev  import chebyshev_data

import numpy as np
import pylab as plt
from copy import deepcopy



def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------     
    vehicle                                     = SUAVE.Vehicle()
    vehicle.tag                                 = 'Vahana'
    vehicle.configuration                       = 'eVTOL'
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff             = 2250. * Units.lb 
    vehicle.mass_properties.operating_empty     = 2250. * Units.lb
    vehicle.mass_properties.max_takeoff         = 2250. * Units.lb
    vehicle.mass_properties.center_of_gravity   = [[ 2.0144,   0.  ,  0.]]
                                                
    vehicle.reference_area                      = 10.58275476  
    vehicle.envelope.ultimate_load              = 5.7
    vehicle.envelope.limit_load                 = 3.     
    
    # ------------------------------------------------------    
    # WINGS    
    # ------------------------------------------------------  
    wing                          = SUAVE.Components.Wings.Main_Wing()
    wing.tag                      = 'canard_wing'  
    wing.aspect_ratio             = 11.37706641  
    wing.sweeps.quarter_chord     = 0.0
    wing.thickness_to_chord       = 0.18  
    wing.taper                    = 1.  
    wing.spans.projected          = 6.65 
    wing.chords.root              = 0.95 
    wing.total_length             = 0.95   
    wing.chords.tip               = 0.95 
    wing.chords.mean_aerodynamic  = 0.95   
    wing.dihedral                 = 0.0  
    wing.areas.reference          = 6.31  
    wing.areas.wetted             = 12.635  
    wing.areas.exposed            = 12.635  
    wing.twists.root              = 0.  
    wing.twists.tip               = 0.  
    wing.origin                   = [[0.1,  0.0 , 0.5]]  
    wing.aerodynamic_center       = [0., 0., 0.]     
    wing.winglet_fraction         = 0.0  
    wing.symmetric                = True        
                                                
    # add to vehicle                            
    vehicle.append_component(wing)              
                                                 
    wing                          = SUAVE.Components.Wings.Main_Wing()
    wing.tag                      = 'main_wing'  
    wing.aspect_ratio             = 11.37706641  
    wing.sweeps.quarter_chord     = 0.0
    wing.thickness_to_chord       = 0.18  
    wing.taper                    = 1.  
    wing.spans.projected          = 6.65 
    wing.chords.root              = 0.95 
    wing.total_length             = 0.95   
    wing.chords.tip               = 0.95 
    wing.chords.mean_aerodynamic  = 0.95   
    wing.dihedral                 = 0.0  
    wing.areas.reference          = 6.31  
    wing.areas.wetted             = 12.635  
    wing.areas.exposed            = 12.635  
    wing.twists.root              = 0.  
    wing.twists.tip               = 0.  
    wing.origin                   = [[ 5.138, 0.0 ,1.54 ]] 
    wing.aerodynamic_center       = [0., 0., 0.]     
    wing.winglet_fraction         = 0.0  
    wing.symmetric                = True 
    
    # add to vehicle 
    vehicle.append_component(wing)   
    
        
    # ------------------------------------------------------    
    # FUSELAGE    
    # ------------------------------------------------------    
    # FUSELAGE PROPERTIES                       
    fuselage                                    = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage' 
    fuselage.seats_abreast                      = 0.  
    fuselage.seat_pitch                         = 1.  
    fuselage.fineness.nose                      = 1.5 
    fuselage.fineness.tail                      = 4.0 
    fuselage.lengths.nose                       = 1.7   
    fuselage.lengths.tail                       = 2.7 
    fuselage.lengths.cabin                      = 1.7  
    fuselage.lengths.total                      = 6.1  
    fuselage.width                              = 1.15  
    fuselage.heights.maximum                    =  1.7 
    fuselage.heights.at_quarter_length          = 1.2  
    fuselage.heights.at_wing_root_quarter_chord = 1.7  
    fuselage.heights.at_three_quarters_length   = 0.75 
    fuselage.areas.wetted                       = 12.97989862  
    fuselage.areas.front_projected              = 1.365211404  
    fuselage.effective_diameter                 = 1.318423736  
    fuselage.differential_pressure              = 0.  
    
    # Segment  
    segment                                     = SUAVE.Components.Fuselages.Segment() 
    segment.tag                                 = 'segment_1'  
    segment.origin                              = [0., 0. ,0.]  
    segment.percent_x_location                  = 0.  
    segment.percent_z_location                  = 0.  
    segment.height                              = 0.09  
    segment.width                               = 0.23473  
    segment.length                              = 0.  
    segment.effective_diameter                  = 0. 
    fuselage.Segments.append(segment)             
                                                
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_2'  
    segment.origin                              = [0.97675, 0. ,0.21977]  
    segment.percent_x_location                  = 0.97675/6.1 
    segment.percent_z_location                  = 0.21977/1.19622 
    segment.height                              = 0.9027  
    segment.width                               = 1.01709  
    fuselage.Segments.append(segment)             
                                                
 
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_3'  
    segment.origin                              = [ 1.93556, 0. ,  0.39371]  
    segment.percent_x_location                  = 1.93556/6.1 
    segment.percent_z_location                  = 0.39371/1.19622
    segment.height                              = 1.30558   
    segment.width                               = 1.38871  
    fuselage.Segments.append(segment)             
 
   
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_4'  
    segment.origin                              = [ 3.44137, 0. , 0.57143]  
    segment.percent_x_location                  = 3.44137/6.1 
    segment.percent_z_location                  = 0.57143/1.19622
    segment.height                              = 1.52588 
    segment.width                               = 1.47074 
    fuselage.Segments.append(segment)             
                                                
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_5'  
    segment.origin                              = [ 4.61031, 0. , 0.81577]  
    segment.percent_x_location                  = 4.61031/6.1
    segment.percent_z_location                  = 0.81577/1.19622 
    segment.height                              = 1.14788 
    segment.width                               = 1.11463  
    fuselage.Segments.append(segment)              
                                                
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_6'  
    segment.origin                              = [ 6.1, 0. ,1.19622]  
    segment.percent_x_location                  = 1. 
    segment.percent_z_location                  = 1.  
    segment.height                              = 0.31818 
    segment.width                               = 0.23443  
    fuselage.Segments.append(segment)             
    
    # add to vehicle
    vehicle.append_component(fuselage)    
    
   
    #------------------------------------------------------------------
    # PROPULSOR
    #------------------------------------------------------------------
    net                   = Vectored_Thrust()    
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
    bat                                                 = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.specific_energy                                 = 300. * Units.Wh/Units.kg
    bat.resistance                                      = 0.005
    bat.max_voltage                                     = net.voltage 
    bat.mass_properties.mass                            = 350. * Units.kg
    initialize_from_mass(bat, bat.mass_properties.mass)
    net.battery                                         = bat

    #------------------------------------------------------------------
    # Design Rotors  
    #------------------------------------------------------------------ 
    # atmosphere conditions 
    speed_of_sound               = 340
    rho                          = 1.22 
    fligth_CL                    = 0.75
    AR                           = vehicle.wings.main_wing.aspect_ratio
    Cd0                          = 0.06
    Cdi                          = fligth_CL**2/(np.pi*AR*0.98)
    Cd                           = Cd0 + Cdi   
                                 
    # Create propeller geometry  
    rot                          = SUAVE.Components.Energy.Converters.Rotor() 
    rot.y_pitch                  = 1.850
    rot.tip_radius               = 0.8875  
    rot.hub_radius               = 0.1 
    rot.disc_area                = np.pi*(rot.tip_radius**2)   
    rot.design_tip_mach          = 0.5
    rot.number_blades            = 3  
    rot.freestream_velocity      = 10     
    rot.angular_velocity         = rot.design_tip_mach*speed_of_sound/rot.tip_radius      
    rot.design_Cl                = 0.7
    rot.design_altitude          = 500 * Units.feet                  
    Lift                         = vehicle.mass_properties.takeoff*9.81  
    rot.design_thrust            = (Lift * 1.5 )/net.number_of_engines 
    rot.induced_hover_velocity   = np.sqrt(Lift/(2*rho*rot.disc_area*net.number_of_engines))                     
    rot                          = propeller_design(rot)  
    rot.rotation                 = [1,1,1,1,1,1,1,1]
                                 
    # Front Rotors Locations 
    rot_front                    = Data()
    rot_front.origin             =  [[0.0 , 1.347 ,0.5 ]]
    rot_front.symmetric          = True
    rot_front.x_pitch_count      = 1
    rot_front.y_pitch_count      = 2     
    rot_front.y_pitch            = 1.85   
                                 
    # populating rotors on one side of wing
    if rot_front.y_pitch_count > 1 :
        for n in range(rot_front.y_pitch_count):
            if n == 0:
                continue
            for i in range(len(rot_front.origin)):
                propeller_origin = [rot_front.origin[i][0] , rot_front.origin[i][1] +  n*rot_front.y_pitch ,rot_front.origin[i][2]]
                rot_front.origin.append(propeller_origin)   
   
                 
    # populating rotors on the other side of the vehicle   
    if rot_front.symmetric : 
        for n in range(len(rot_front.origin)):
            propeller_origin = [rot_front.origin[n][0] , -rot_front.origin[n][1] ,rot_front.origin[n][2] ]
            rot_front.origin.append(propeller_origin) 
      
    # Rear Rotors Locations 
    rot_rear               = Data()
    rot_rear.origin        =  [[ 5.138, 1.347 ,1.54 ]]  
    rot_rear.symmetric     = True
    rot_rear.x_pitch_count = 1
    rot_rear.y_pitch_count = 2     
    rot_rear.y_pitch       = 1.85                   
    # populating rotors on one side of wing
    if rot_rear.y_pitch_count > 1 :
        for n in range(rot_rear.y_pitch_count):
            if n == 0:
                continue
            for i in range(len(rot_rear.origin)):
                propeller_origin = [rot_rear.origin[i][0] , rot_rear.origin[i][1] +  n*rot_rear.y_pitch ,rot_rear.origin[i][2]]
                rot_rear.origin.append(propeller_origin)   


    # populating rotors on the other side of the vehicle   
    if rot_rear.symmetric : 
        for n in range(len(rot_rear.origin)):
            propeller_origin = [rot_rear.origin[n][0] , -rot_rear.origin[n][1] ,rot_rear.origin[n][2] ]
            rot_rear.origin.append(propeller_origin) 
      
    # Assign all rotors (front and rear) to network
    rot.origin = rot_front.origin + rot_rear.origin   
    
    # append rotors to vehicle     
    net.rotor = rot
    
    # Motor
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    # Propeller (Thrust) motor
    motor                      = SUAVE.Components.Energy.Converters.Motor()
    motor.mass_properties.mass = 9. * Units.kg
    motor.origin               = rot_front.origin + rot_rear.origin  
    motor.efficiency           = 0.935
    motor.gear_ratio           = 1. 
    motor.gearbox_efficiency   = 1. # Gear box efficiency        
    motor.nominal_voltage      = bat.max_voltage *3/4  
    motor.propeller_radius     = rot.tip_radius 
    motor.no_load_current      = 2.0 
    motor                      = size_optimal_motor(motor,rot) 
    net.motor                  = motor 
    
    vehicle.append_component(net) 
     
    
    # Add extra drag sources from motors, props, and landing gear. All of these hand measured 
    motor_height                     = .25 * Units.feet
    motor_width                      =  1.6 * Units.feet 
    propeller_width                  = 1. * Units.inches
    propeller_height                 = propeller_width *.12 
    main_gear_width                  = 1.5 * Units.inches
    main_gear_length                 = 2.5 * Units.feet 
    nose_gear_width                  = 2. * Units.inches
    nose_gear_length                 = 2. * Units.feet 
    nose_tire_height                 = (0.7 + 0.4) * Units.feet
    nose_tire_width                  = 0.4 * Units.feet  
    main_tire_height                 = (0.75 + 0.5) * Units.feet
    main_tire_width                  = 4. * Units.inches 
    total_excrescence_area_spin      = 12.*motor_height*motor_width + 2.* main_gear_length*main_gear_width \
                                       + nose_gear_width*nose_gear_length + 2 * main_tire_height*main_tire_width\
                                       + nose_tire_height*nose_tire_width 
    total_excrescence_area_no_spin   = total_excrescence_area_spin + 12*propeller_height*propeller_width  
    vehicle.excrescence_area_no_spin = total_excrescence_area_no_spin 
    vehicle.excrescence_area_spin    = total_excrescence_area_spin  
    # append motor origin spanwise locations onto wing data structure 
    motor_origins_front              = np.array(rot_front.origin)
    
    vehicle.wings['canard_wing'].motor_spanwise_locations = np.multiply(
        0.19 ,motor_origins_front[:,1])
    motor_origins_rear = np.array(rot_rear.origin)
    vehicle.wings['main_wing'].motor_spanwise_locations = np.multiply(
        0.19 ,motor_origins_rear[:,1])    
    
    return vehicle


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
    configs                                     = SUAVE.Components.Configs.Config.Container() 
    base_config                                 = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag                             = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Hover Configuration
    # ------------------------------------------------------------------
    config                                            = SUAVE.Components.Configs.Config(base_config)
    config.tag                                        = 'hover'
    config.propulsors.vectored_thrust.thrust_angle    = 90.0 * Units.degrees
    config.propulsors.vectored_thrust.pitch_command   = 0.  * Units.degrees    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Hover Climb Configuration
    # ------------------------------------------------------------------
    config                                            = SUAVE.Components.Configs.Config(base_config)
    config.tag                                        = 'hover_climb'
    config.propulsors.vectored_thrust.thrust_angle    = 90.0 * Units.degrees
    config.propulsors.vectored_thrust.pitch_command   = -5.  * Units.degrees    
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    config                                            = SUAVE.Components.Configs.Config(base_config)
    config.tag                                        = 'cruise' 
    config.propulsors.vectored_thrust.thrust_angle    =  0. * Units.degrees
    config.propulsors.vectored_thrust.pitch_command   = 10.  * Units.degrees  
    configs.append(config)  
    
    return configs

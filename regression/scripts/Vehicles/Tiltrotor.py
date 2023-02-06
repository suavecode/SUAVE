# Tiltrotor.py
#
# Created:  Mar 2022, R. Erhard

""" setup file for a tiltrotor configuration based on the geometry of 
    the X57-Maxwell Modification 2 Electric Aircraft
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import MARC
from MARC.Core import Units
from MARC.Components.Energy.Networks.Battery_Electric_Rotor     import Battery_Electric_Rotor
from MARC.Methods.Propulsion                           import propeller_design
from MARC.Methods.Power.Battery.Sizing                 import initialize_from_mass
from MARC.Methods.Propulsion.electric_motor_sizing     import size_optimal_motor
from MARC.Methods.Geometry.Two_Dimensional.Planform    import wing_segmented_planform

import numpy as np 
from copy import deepcopy

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    vehicle = MARC.Vehicle()
    vehicle.tag = 'Tiltrotor'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------
    vehicle.mass_properties.max_takeoff   = 2550. * Units.pounds
    vehicle.mass_properties.takeoff       = 2550. * Units.pounds
    vehicle.mass_properties.max_zero_fuel = 2550. * Units.pounds 
    vehicle.envelope.ultimate_load        = 5.7
    vehicle.envelope.limit_load           = 3.8 
    vehicle.reference_area                = 14.76
    vehicle.passengers                    = 4
    vehicle.systems.control               = "fully powered"
    vehicle.systems.accessories           = "commuter"    
    
    cruise_speed                          = 135.*Units['mph']    
    altitude                              = 2500. * Units.ft
    atmo                                  = MARC.Analyses.Atmospheric.US_Standard_1976()
    freestream                            = atmo.compute_values (0.)
    freestream0                           = atmo.compute_values (altitude)
    mach_number                           = (cruise_speed/freestream.speed_of_sound)[0][0] 
    vehicle.design_dynamic_pressure       = ( .5 *freestream0.density*(cruise_speed*cruise_speed))[0][0]
    vehicle.design_mach_number            =  mach_number
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------    
    wing                                  = MARC.Components.Wings.Main_Wing()
    wing.tag                              = 'main_wing' 
    wing.sweeps.quarter_chord             = 0.0 * Units.deg
    wing.thickness_to_chord               = 0.12
    wing.areas.reference                  = 14.76
    wing.spans.projected                  = 11.4 
    wing.chords.root                      = 1.46
    wing.chords.tip                       = 0.92
    wing.chords.mean_aerodynamic          = 1.19
    wing.taper                            = wing.chords.root/wing.chords.tip 
    wing.aspect_ratio                     = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root                      = 3.0 * Units.degrees
    wing.twists.tip                       = 0.0 * Units.degrees 
    wing.origin                           = [[2.93, 0., 1.01]]
    wing.aerodynamic_center               = [3., 0., 1.01] 
    wing.vertical                         = False
    wing.symmetric                        = True
    wing.high_lift                        = True 
    wing.winglet_fraction                 = 0.0  
    wing.dynamic_pressure_ratio           = 1.0  
    airfoil                               = MARC.Components.Airfoils.Airfoil()
    airfoil.coordinate_file               = '../Vehicles/Airfoils/NACA_63_412.txt'
    
    cg_x = wing.origin[0][0] + 0.25*wing.chords.mean_aerodynamic
    cg_z = wing.origin[0][2] - 0.2*wing.chords.mean_aerodynamic
    vehicle.mass_properties.center_of_gravity = [[cg_x,   0.  ,  cg_z ]]  # SOURCE: Design and aerodynamic analysis of a twin-engine commuter aircraft

    # Wing Segments
    segment                               = MARC.Components.Wings.Segment()
    segment.tag                           = 'inboard'
    segment.percent_span_location         = 0.0 
    segment.twist                         = 3. * Units.degrees   
    segment.root_chord_percent            = 1. 
    segment.dihedral_outboard             = 0.  
    segment.sweeps.quarter_chord          = 0.
    segment.thickness_to_chord            = 0.12
    segment.append_airfoil(airfoil)
    wing.append_segment(segment)

    segment                               = MARC.Components.Wings.Segment()
    segment.tag                           = 'outboard'
    segment.percent_span_location         = 0.5438
    segment.twist                         = 2.* Units.degrees 
    segment.root_chord_percent            = 1. 
    segment.dihedral_outboard             = 0. 
    segment.sweeps.quarter_chord          = 0.
    segment.thickness_to_chord            = 0.12 
    segment.append_airfoil(airfoil)
    wing.append_segment(segment)

    segment                               = MARC.Components.Wings.Segment()
    segment.tag                           = 'tip'
    segment.percent_span_location         = 1.
    segment.twist                         = 0. * Units.degrees 
    segment.root_chord_percent            = 0.630
    segment.dihedral_outboard             = 0.
    segment.sweeps.quarter_chord          = 0.
    segment.thickness_to_chord            = 0.12
    segment.append_airfoil(airfoil)
    wing.append_segment(segment)    
    
    # Fill out more segment properties automatically
    wing = wing_segmented_planform(wing)           
    
    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------       
    wing                                  = MARC.Components.Wings.Wing()
    wing.tag                              = 'horizontal_stabilizer' 
    wing.sweeps.quarter_chord             = 0.0 * Units.deg
    wing.thickness_to_chord               = 0.12
    wing.areas.reference                  = 2.540 
    wing.spans.projected                  = 3.3  * Units.meter 
    wing.sweeps.quarter_chord             = 0 * Units.deg 
    wing.chords.root                      = 0.769 * Units.meter 
    wing.chords.tip                       = 0.769 * Units.meter 
    wing.chords.mean_aerodynamic          = 0.769 * Units.meter  
    wing.taper                            = 1. 
    wing.aspect_ratio                     = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root                      = 0.0 * Units.degrees
    wing.twists.tip                       = 0.0 * Units.degrees 
    wing.origin                           = [[7.7, 0., 0.25]]
    wing.aerodynamic_center               = [7.8, 0., 0.25] 
    wing.vertical                         = False
    wing.winglet_fraction                 = 0.0  
    wing.symmetric                        = True
    wing.high_lift                        = False 
    wing.dynamic_pressure_ratio           = 0.9

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------ 
    wing                                  = MARC.Components.Wings.Wing()
    wing.tag                              = 'vertical_stabilizer'     
    wing.sweeps.quarter_chord             = 25. * Units.deg
    wing.thickness_to_chord               = 0.12
    wing.areas.reference                  = 2.258 * Units['meters**2']  
    wing.spans.projected                  = 1.854   * Units.meter  
    wing.chords.root                      = 1.6764 * Units.meter 
    wing.chords.tip                       = 0.6858 * Units.meter 
    wing.chords.mean_aerodynamic          = 1.21   * Units.meter 
    wing.taper                            = wing.chords.tip/wing.chords.root 
    wing.aspect_ratio                     = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root                      = 0.0 * Units.degrees
    wing.twists.tip                       = 0.0 * Units.degrees 
    wing.origin                           = [[6.75 ,0, 0.623]]
    wing.aerodynamic_center               = [0.508 ,0,0]  
    wing.vertical                         = True 
    wing.symmetric                        = False
    wing.t_tail                           = False
    wing.winglet_fraction                 = 0.0  
    wing.dynamic_pressure_ratio           = 1.0

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    fuselage = MARC.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage'
    fuselage.seats_abreast                      = 2.
    fuselage.fineness.nose                      = 1.6
    fuselage.fineness.tail                      = 2.
    fuselage.lengths.nose                       = 60.  * Units.inches
    fuselage.lengths.tail                       = 161. * Units.inches
    fuselage.lengths.cabin                      = 105. * Units.inches
    fuselage.lengths.total                      = 332.2* Units.inches
    fuselage.lengths.fore_space                 = 0.
    fuselage.lengths.aft_space                  = 0.
    fuselage.width                              = 42. * Units.inches
    fuselage.heights.maximum                    = 62. * Units.inches
    fuselage.heights.at_quarter_length          = 62. * Units.inches
    fuselage.heights.at_three_quarters_length   = 62. * Units.inches
    fuselage.heights.at_wing_root_quarter_chord = 23. * Units.inches
    fuselage.areas.side_projected               = 8000.  * Units.inches**2.
    fuselage.areas.wetted                       = 30000. * Units.inches**2.
    fuselage.areas.front_projected              = 42.* 62. * Units.inches**2.
    fuselage.effective_diameter                 = 50. * Units.inches 

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_0'
    segment.percent_x_location                  = 0
    segment.percent_z_location                  = 0
    segment.height                              = 0.01
    segment.width                               = 0.01
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_1'
    segment.percent_x_location                  = 0.007279116466
    segment.percent_z_location                  = 0.002502014453
    segment.height                              = 0.1669064748
    segment.width                               = 0.2780205877
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_2'
    segment.percent_x_location                  = 0.01941097724
    segment.percent_z_location                  = 0.001216095397
    segment.height                              = 0.3129496403
    segment.width                               = 0.4365777215
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_3'
    segment.percent_x_location                  = 0.06308567604
    segment.percent_z_location                  = 0.007395489231
    segment.height                              = 0.5841726619
    segment.width                               = 0.6735119903
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_4'
    segment.percent_x_location                  = 0.1653761217
    segment.percent_z_location                  = 0.02891281352
    segment.height                              = 1.064028777
    segment.width                               = 1.067200529
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_5'
    segment.percent_x_location                  = 0.2426372155
    segment.percent_z_location                  = 0.04214148761
    segment.height                              = 1.293766653
    segment.width                               = 1.183058255
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_6'
    segment.percent_x_location                  = 0.2960174029
    segment.percent_z_location                  = 0.04705241831
    segment.height                              = 1.377026712
    segment.width                               = 1.181540054
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_7'
    segment.percent_x_location                  = 0.3809404284
    segment.percent_z_location                  = 0.05313580461
    segment.height                              = 1.439568345
    segment.width                               = 1.178218989
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_8'
    segment.percent_x_location                  = 0.5046854083
    segment.percent_z_location                  = 0.04655492473
    segment.height                              = 1.29352518
    segment.width                               = 1.054390707
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_9'
    segment.percent_x_location                  = 0.6454149933
    segment.percent_z_location                  = 0.03741966266
    segment.height                              = 0.8971223022
    segment.width                               = 0.8501926505
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_10'
    segment.percent_x_location                  = 0.985107095
    segment.percent_z_location                  = 0.04540283436
    segment.height                              = 0.2920863309
    segment.width                               = 0.2012565415
    fuselage.Segments.append(segment)

    # Segment
    segment                                     = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                                 = 'segment_11'
    segment.percent_x_location                  = 1
    segment.percent_z_location                  = 0.04787575562
    segment.height                              = 0.1251798561
    segment.width                               = 0.1206021048
    fuselage.Segments.append(segment)

    # add to vehicle
    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    #   Nacelles
    # ------------------------------------------------------------------ 
    nacelle                = MARC.Components.Nacelles.Nacelle()
    nacelle.tag            = 'nacelle_1'
    nacelle.length         = 2.5
    nacelle.diameter       = 42 * Units.inches
    nacelle.areas.wetted   = 0.01*(2*np.pi*0.01/2)
    nacelle.origin         = [[2.,5.7,1.0]]
    nacelle.flow_through   = False  
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_1'
    nac_segment.percent_x_location = 0.0  
    nac_segment.height             = 0.0
    nac_segment.width              = 0.0
    nacelle.append_segment(nac_segment)   
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_2'
    nac_segment.percent_x_location = 0.1  
    nac_segment.height             = 0.5
    nac_segment.width              = 0.65
    nacelle.append_segment(nac_segment)   
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_3'
    nac_segment.percent_x_location = 0.3  
    nac_segment.height             = 0.52
    nac_segment.width              = 0.7
    nacelle.append_segment(nac_segment)  
     
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_4'
    nac_segment.percent_x_location = 0.5  
    nac_segment.height             = 0.5
    nac_segment.width              = 0.65
    nacelle.append_segment(nac_segment)  
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_5'
    nac_segment.percent_x_location = 0.7 
    nac_segment.height             = 0.4
    nac_segment.width              = 0.6
    nacelle.append_segment(nac_segment)   
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_6'
    nac_segment.percent_x_location = 0.9 
    nac_segment.height             = 0.3
    nac_segment.width              = 0.5
    nacelle.append_segment(nac_segment)  
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_7'
    nac_segment.percent_x_location = 1.0  
    nac_segment.height             = 0.0
    nac_segment.width              = 0.0
    nacelle.append_segment(nac_segment)    
    
    vehicle.append_component(nacelle)  

    nacelle_2          = deepcopy(nacelle)
    nacelle_2.tag      = 'nacelle_2'
    nacelle_2.origin   = [[2.,-5.7,1.0]]
    vehicle.append_component(nacelle_2)    
    
    #---------------------------------------------------------------------------------------------
    # DEFINE PROPELLER
    #---------------------------------------------------------------------------------------------
    # build network
    net = Battery_Electric_Rotor()   
    net.rotor_group_indexes      = [0,0]
    net.motor_group_indexes      = [0,0] 
    net.esc_group_indexes        = [0,0] 

    # Component 1 the ESC
    esc_1            = MARC.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc_1.efficiency = 0.95 
    esc_1.tag        = 'esc_1'
    net.electronic_speed_controllers.append(esc_1) 

    esc_2            = MARC.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc_2.efficiency = 0.95 
    esc_2.tag        = 'esc_1'
    net.electronic_speed_controllers.append(esc_2)
    

    Hover_Load = vehicle.mass_properties.takeoff * 9.81
    
    # Component 2 the Propeller 
    prop                                      = MARC.Components.Energy.Converters.Propeller()
    prop.tag                                  = 'propeller_1'
    prop.tip_radius                           = 1.5
    prop.number_of_blades                     = 3.0
    prop.hub_radius                           = 0.15*prop.tip_radius
    prop.cruise.design_Cl                     = 0.8
    prop.cruise.design_freestream_velocity    = 90.   * Units.mph
    prop.cruise.design_tip_mach               = 0.6
    prop.cruise.design_angular_velocity       = prop.cruise.design_tip_mach*340 /prop.tip_radius
    prop.cruise.design_altitude               = 3000. * Units.feet  
    prop.cruise.design_thrust                 = (1.1*Hover_Load)/2
    prop.origin                               = [[2.,5.7,0.784]]
    prop.rotation                             = -1
    prop.sol_tolerance                        = 1e-4
    prop.symmetry                             = True
    prop.variable_pitch                       = True 
    airfoil                                   = MARC.Components.Airfoils.Airfoil()   
    airfoil.coordinate_file                   = '../Vehicles/Airfoils/NACA_4412.txt'
    airfoil.polar_files                       = ['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                              '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                              '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                              '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                              '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ] 
    prop.append_airfoil(airfoil)              
    prop.airfoil_polar_stations                = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    prop                         = propeller_design(prop)

    prop_left = deepcopy(prop)
    prop_left.tag = 'propeller_2' 
    prop_left.origin   = [[2.,-5.7,0.784]]
    prop_left.rotation = 1
    
    net.rotors.append(prop)
    net.rotors.append(prop_left)


    # Component 3 the Battery
    bat = MARC.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650()
    bat.mass_properties.mass = 500. * Units.kg  
    bat.pack.max_voltage     = 400. 
    initialize_from_mass(bat)
    net.battery              = bat
    net.voltage              = bat.pack.max_voltage

    # Component 4 Miscellaneous Systems
    sys = MARC.Components.Systems.System()
    sys.mass_properties.mass = 5 # kg
 
    # Component 5 the Motor  
    motor                         = MARC.Components.Energy.Converters.Motor()
    motor.efficiency              = 0.95
    motor.gearbox_efficiency      = 1.
    motor.origin                  = [[2.,  5.7, 0.95]]
    motor.nominal_voltage         = bat.pack.max_voltage*0.8 
    motor.no_load_current         = 0.1 
    motor.rotor_radius            = prop.tip_radius
    motor.design_torque           = prop.cruise.design_torque
    motor.angular_velocity        = prop.cruise.design_angular_velocity/motor.gear_ratio
    motor                         = size_optimal_motor(motor) 
    motor.mass_properties.mass    = 10. * Units.kg 
    
    # append right motor
    net.motors.append(motor)
    
    # append left motor 
    motor_left = deepcopy(motor)
    motor_left.origin = [[2., -5.7, 0.95]] 
    net.motors.append(motor_left) 

    # Component 6 the Payload
    payload = MARC.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 10. # Watts
    payload.mass_properties.mass = 1.0 * Units.kg
    net.payload                  = payload

    # Component 7 the Avionics
    avionics = MARC.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 20. # Watts
    net.avionics        = avionics

    # add the solar network to the vehicle
    vehicle.append_component(net)

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    return vehicle

# ---------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = MARC.Components.Configs.Config.Container()

    base_config = MARC.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    base_config.networks.battery_electric_rotor.pitch_command = 0
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Hover Configuration
    # ------------------------------------------------------------------
    config = MARC.Components.Configs.Config(base_config)
    config.tag = 'hover_climb'
    vector_angle = 90.0 * Units.degrees
    config.networks.battery_electric_rotor.rotors['propeller_1'].inputs.y_axis_rotation=  vector_angle
    config.networks.battery_electric_rotor.rotors['propeller_2'].inputs.y_axis_rotation =  vector_angle 
    config.networks.battery_electric_rotor.pitch_command = 0. * Units.degrees
    configs.append(config)

    # ------------------------------------------------------------------
    #   Transition 1 Configuration
    # ------------------------------------------------------------------
    config = MARC.Components.Configs.Config(base_config)
    config.tag = 'transition_1'
    config.networks.battery_electric_rotor.pitch_command = 0. * Units.degrees
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    config = MARC.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    vector_angle = 0.0 * Units.degrees
    config.networks.battery_electric_rotor.rotors['propeller_1'].inputs.y_axis_rotation =  vector_angle
    config.networks.battery_electric_rotor.rotors['propeller_2'].inputs.y_axis_rotation =  vector_angle  
    configs.append(config)
        
    # done!
    return configs

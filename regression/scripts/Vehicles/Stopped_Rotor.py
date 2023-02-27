# Stopped_Rotor_CRM.py
#
# Created: May 2019, M Clarke
#          Sep 2020, M. Clarke

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import MARC
from MARC.Core import Units, Data
import copy 
from MARC.Components.Energy.Networks.Battery_Electric_Rotor              import Battery_Electric_Rotor
from MARC.Methods.Power.Battery.Sizing                                   import initialize_from_mass
from MARC.Methods.Performance.estimate_cruise_drag                       import estimate_cruise_drag
from MARC.Methods.Propulsion.electric_motor_sizing                       import size_from_mass , size_optimal_motor
from MARC.Methods.Propulsion                                             import propeller_design
from MARC.Methods.Weights.Buildups.eVTOL.empty                           import empty
from MARC.Methods.Center_of_Gravity.compute_component_centers_of_gravity import compute_component_centers_of_gravity
from MARC.Methods.Geometry.Two_Dimensional.Planform import wing_segmented_planform


import numpy as np
import pylab as plt
from copy import deepcopy

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    vehicle               = MARC.Vehicle()
    vehicle.tag           = 'Lift_Cruise_CRM'
    vehicle.configuration = 'eVTOL'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------
    # mass properties
    vehicle.mass_properties.takeoff           = 2450. * Units.lb
    vehicle.mass_properties.operating_empty   = 2250. * Units.lb               # Approximate
    vehicle.mass_properties.max_takeoff       = 2450. * Units.lb               # Approximate
    vehicle.mass_properties.max_payload       = 200.  * Units.lb
    vehicle.mass_properties.center_of_gravity = [[2.0144,   0.  ,  0. ]] # Approximate

    # basic parameters
    vehicle.reference_area         = 10.76
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load    = 3.

    # ------------------------------------------------------------------
    # WINGS
    # ------------------------------------------------------------------
    # WING PROPERTIES
    wing                          = MARC.Components.Wings.Main_Wing()
    wing.tag                      = 'main_wing'
    wing.aspect_ratio             = 10.76
    wing.sweeps.quarter_chord     = 0.0  * Units.degrees
    wing.thickness_to_chord       = 0.18
    wing.taper                    = 1.
    wing.spans.projected          = 35.0   * Units.feet
    wing.chords.root              = 3.25   * Units.feet
    wing.total_length             = 3.25   * Units.feet
    wing.chords.tip               = 3.25   * Units.feet
    wing.chords.mean_aerodynamic  = 3.25   * Units.feet
    wing.dihedral                 = 1.0    * Units.degrees
    wing.areas.reference          = 113.75 * Units.feet**2
    wing.areas.wetted             = 227.5  * Units.feet**2
    wing.areas.exposed            = 227.5  * Units.feet**2
    wing.twists.root              = 4.0    * Units.degrees
    wing.twists.tip               = 0.0    * Units.degrees
    wing.origin                   = [[1.5, 0., 0. ]]
    wing.aerodynamic_center       = [1.975 , 0., 0.]
    wing.winglet_fraction         = 0.0
    wing.symmetric                = True
    wing.vertical                 = False

    # Segment
    segment                       = MARC.Components.Wings.Segment()
    segment.tag                   = 'Section_1'
    segment.percent_span_location = 0.
    segment.twist                 = 0.
    segment.root_chord_percent    = 1.5
    segment.dihedral_outboard     = 1.0 * Units.degrees
    segment.sweeps.quarter_chord  = 8.5 * Units.degrees
    segment.thickness_to_chord    = 0.18
    wing.Segments.append(segment)

    # Segment
    segment                       = MARC.Components.Wings.Segment()
    segment.tag                   = 'Section_2'
    segment.percent_span_location = 0.227
    segment.twist                 = 0.
    segment.root_chord_percent    = 1.
    segment.dihedral_outboard     = 1.0  * Units.degrees
    segment.sweeps.quarter_chord  = 0.0  * Units.degrees
    segment.thickness_to_chord    = 0.12
    wing.Segments.append(segment)

    # Segment
    segment                       = MARC.Components.Wings.Segment()
    segment.tag                   = 'Section_3'
    segment.percent_span_location = 1.0
    segment.twist                 = 0.
    segment.root_chord_percent    = 1.0
    segment.dihedral_outboard     = 1.0  * Units.degrees
    segment.sweeps.quarter_chord  = 0.0 * Units.degrees
    segment.thickness_to_chord    = 0.12
    wing.Segments.append(segment)
    
    # Fill out more segment properties automatically
    wing = wing_segmented_planform(wing)        

    # add to vehicle
    vehicle.append_component(wing)

    # WING PROPERTIES
    wing                          = MARC.Components.Wings.Wing()
    wing.tag                      = 'horizontal_tail'
    wing.aspect_ratio             = 4.0
    wing.sweeps.quarter_chord     = 0.0
    wing.thickness_to_chord       = 0.12
    wing.taper                    = 1.0
    wing.spans.projected          = 8.0 * Units.feet
    wing.chords.root              = 2.0 * Units.feet
    wing.total_length             = 2.0 * Units.feet
    wing.chords.tip               = 2.0 * Units.feet
    wing.chords.mean_aerodynamic  = 2.0 * Units.feet
    wing.dihedral                 = 0.  * Units.degrees
    wing.areas.reference          = 16.0  * Units.feet**2
    wing.areas.wetted             = 32.0  * Units.feet**2
    wing.areas.exposed            = 32.0  * Units.feet**2
    wing.twists.root              = 0. * Units.degrees
    wing.twists.tip               = 0. * Units.degrees
    wing.origin                   = [[4.0 , 0.0 , 0.205 ]]
    wing.aerodynamic_center       = [4.2 ,  0.,  0.]
    wing.symmetric                = True

    # add to vehicle
    vehicle.append_component(wing)


    # WING PROPERTIES
    wing                          = MARC.Components.Wings.Wing()
    wing.tag                      = 'vertical_tail_1'
    wing.aspect_ratio             = 2.
    wing.sweeps.quarter_chord     = 20.0 * Units.degrees
    wing.thickness_to_chord       = 0.12
    wing.taper                    = 0.5
    wing.spans.projected          = 3.0 * Units.feet
    wing.chords.root              = 2.0 * Units.feet
    wing.total_length             = 2.0 * Units.feet
    wing.chords.tip               = 1.0 * Units.feet
    wing.chords.mean_aerodynamic  = 1.5 * Units.feet
    wing.areas.reference          = 4.5 * Units.feet**2
    wing.areas.wetted             = 9.0 * Units.feet**2
    wing.areas.exposed            = 9.0 * Units.feet**2
    wing.twists.root              = 0. * Units.degrees
    wing.twists.tip               = 0. * Units.degrees
    wing.origin                   = [[4.0, 4.0*0.3048  , 0.205  ]]
    wing.aerodynamic_center       = [4.2,0,0]  
    wing.winglet_fraction         = 0.0
    wing.vertical                 = True
    wing.symmetric                = False

    # add to vehicle
    vehicle.append_component(wing)


    # WING PROPERTIES
    wing                         = MARC.Components.Wings.Wing()
    wing.tag                     = 'vertical_tail_2'
    wing.aspect_ratio            = 2.
    wing.sweeps.quarter_chord    = 20.0 * Units.degrees
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 0.5
    wing.spans.projected         = 3.0 * Units.feet
    wing.chords.root             = 2.0 * Units.feet
    wing.total_length            = 2.0 * Units.feet
    wing.chords.tip              = 1.0 * Units.feet
    wing.chords.mean_aerodynamic = 1.5 * Units.feet
    wing.areas.reference         = 4.5 * Units.feet**2
    wing.areas.wetted            = 9.0 * Units.feet**2
    wing.areas.exposed           = 9.0 * Units.feet**2
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.origin                  = [[4.0, -4.0*0.3048  , 0.205   ]]
    wing.aerodynamic_center      = [4.2,0,0]  
    wing.winglet_fraction        = 0.0
    wing.vertical                = True
    wing.symmetric               = False

    # add to vehicle
    vehicle.append_component(wing)

   # ---------------------------------------------------------------
    # FUSELAGE
    # ---------------------------------------------------------------
    # FUSELAGE PROPERTIES
    fuselage                                    = MARC.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage'
    fuselage.configuration                      = 'Tube_Wing'
    fuselage.seats_abreast                      = 2.
    fuselage.seat_pitch                         = 3.
    fuselage.fineness.nose                      = 0.88
    fuselage.fineness.tail                      = 1.13
    fuselage.lengths.nose                       = 3.2 * Units.feet
    fuselage.lengths.tail                       = 6.4 * Units.feet
    fuselage.lengths.cabin                      = 6.4 * Units.feet
    fuselage.lengths.total                      = 5.1
    fuselage.width                              = 5.85 * Units.feet
    fuselage.heights.maximum                    = 4.65 * Units.feet
    fuselage.heights.at_quarter_length          = 3.75 * Units.feet
    fuselage.heights.at_wing_root_quarter_chord = 4.65 * Units.feet
    fuselage.heights.at_three_quarters_length   = 4.26 * Units.feet
    fuselage.areas.wetted                       = 236. * Units.feet**2
    fuselage.areas.front_projected              = 0.14 * Units.feet**2
    fuselage.effective_diameter                 = 5.85 * Units.feet
    fuselage.differential_pressure              = 0.

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_0'
    segment.percent_x_location        = 0.
    segment.percent_z_location        = -0.267/4.10534
    segment.height                    = 0.1
    segment.width                     = 0.1
    fuselage.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_1'
    segment.percent_x_location        = 0.2579 /4.10534
    segment.percent_z_location        = -0.05881/1.372
    segment.height                    = 0.5201
    segment.width                     = 0.75
    fuselage.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_2'
    segment.percent_x_location        =  0.9939/4.10534
    segment.percent_z_location        =  -0.0446/4.10534
    segment.height                    =  1.18940
    segment.width                     =  1.42045
    fuselage.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_3'
    segment.percent_x_location        =  1.95060 /4.10534
    segment.percent_z_location        =  0
    segment.height                    =  1.37248
    segment.width                     =  1.35312
    fuselage.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_4'
    segment.percent_x_location        = 3.02797/4.10534
    segment.percent_z_location        = 0.25/4.10534
    segment.height                    = 0.6
    segment.width                     = 0.4
    fuselage.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_5'
    segment.percent_x_location        = 1.
    segment.percent_z_location        = 0.42522/4.10534
    segment.height                    = 0.05
    segment.width                     = 0.05
    fuselage.Segments.append(segment)


    # add to vehicle
    vehicle.append_component(fuselage)

   #-------------------------------------------------------------------
    # INNER BOOMS
    #-------------------------------------------------------------------
    long_boom                                    = MARC.Components.Booms.Boom()
    long_boom.tag                                = 'boom_1r'
    long_boom.configuration                      = 'boom'
    long_boom.origin                             = [[0.543,1.630, -0.326]]
    long_boom.seats_abreast                      = 0.
    long_boom.seat_pitch                         = 0.0
    long_boom.fineness.nose                      = 0.950
    long_boom.fineness.tail                      = 1.029
    long_boom.lengths.nose                       = 0.2
    long_boom.lengths.tail                       = 0.2
    long_boom.lengths.cabin                      = 2.5
    long_boom.lengths.total                      = 3.5
    long_boom.width                              = 0.15
    long_boom.heights.maximum                    = 0.15
    long_boom.heights.at_quarter_length          = 0.15
    long_boom.heights.at_three_quarters_length   = 0.15
    long_boom.heights.at_wing_root_quarter_chord = 0.15
    long_boom.areas.wetted                       = 0.018
    long_boom.areas.front_projected              = 0.018
    long_boom.effective_diameter                 = 0.15
    long_boom.differential_pressure              = 0.
    long_boom.symmetric                          = True
    long_boom.index                              = 1

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_1'
    segment.percent_x_location        = 0.
    segment.percent_z_location        = 0.0
    segment.height                    = 0.05
    segment.width                     = 0.05
    long_boom.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_2'
    segment.percent_x_location        = 0.2/ 5.6
    segment.percent_z_location        = 0.
    segment.height                    = 0.15
    segment.width                     = 0.15
    long_boom.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_3'
    segment.percent_x_location        = 5.4/5.6
    segment.percent_z_location        = 0.
    segment.height                    = 0.15
    segment.width                     = 0.15
    long_boom.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_4'
    segment.percent_x_location        = 1.
    segment.percent_z_location        = 0.
    segment.height                    = 0.05
    segment.width                     = 0.05
    long_boom.Segments.append(segment)

    # add to vehicle
    vehicle.append_component(long_boom)

    # add left long boom
    long_boom              = deepcopy(vehicle.booms.boom_1r)
    long_boom.origin[0][1] = -long_boom.origin[0][1]
    long_boom.tag          = 'Boom_1L'
    long_boom.index        = 1
    vehicle.append_component(long_boom)


    #-------------------------------------------------------------------
    # OUTER BOOMS
    #-------------------------------------------------------------------
    short_boom                                    = MARC.Components.Booms.Boom()
    short_boom.tag                                = 'boom_2r'
    short_boom.configuration                      = 'boom'
    short_boom.origin                             = [[0.543,2.826, -0.326]]
    short_boom.seats_abreast                      = 0.
    short_boom.seat_pitch                         = 0.0
    short_boom.fineness.nose                      = 0.950
    short_boom.fineness.tail                      = 1.029
    short_boom.lengths.nose                       = 0.2
    short_boom.lengths.tail                       = 0.2
    short_boom.lengths.cabin                      = 2.0
    short_boom.lengths.total                      = 3.3
    short_boom.width                              = 0.15
    short_boom.heights.maximum                    = 0.15
    short_boom.heights.at_quarter_length          = 0.15
    short_boom.heights.at_three_quarters_length   = 0.15
    short_boom.heights.at_wing_root_quarter_chord = 0.15
    short_boom.areas.wetted                       = 0.018
    short_boom.areas.front_projected              = 0.018
    short_boom.effective_diameter                 = 0.15
    short_boom.differential_pressure              = 0.
    short_boom.y_pitch_count                      = 2
    short_boom.y_pitch                            = 1.196
    short_boom.symmetric                          = True
    short_boom.index                              = 1

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_1'
    segment.percent_x_location        = 0.
    segment.percent_z_location        = 0.0
    segment.height                    = 0.05
    segment.width                     = 0.05
    short_boom.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_2'
    segment.percent_x_location        = 0.2/3.3
    segment.percent_z_location        = 0.
    segment.height                    = 0.15
    segment.width                     = 0.15
    short_boom.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_3'
    segment.percent_x_location        = 3.1/3.3
    segment.percent_z_location        = 0.
    segment.height                    = 0.15
    segment.width                     = 0.15
    short_boom.Segments.append(segment)

    # Segment
    segment                           = MARC.Components.Lofted_Body_Segment.Segment()
    segment.tag                       = 'segment_4'
    segment.percent_x_location        = 1.
    segment.percent_z_location        = 0.
    segment.height                    = 0.05
    segment.width                     = 0.05
    short_boom.Segments.append(segment)

    # add to vehicle
    vehicle.append_component(short_boom)

    # add outer right boom
    short_boom              = deepcopy(vehicle.booms.boom_2r)
    short_boom.origin[0][1] = short_boom.y_pitch + short_boom.origin[0][1]
    short_boom.tag          = 'boom_3r'
    short_boom.index        = 1
    vehicle.append_component(short_boom)

    # add inner left boom
    short_boom              = deepcopy(vehicle.booms.boom_2r)
    short_boom.origin[0][1] = - (short_boom.origin[0][1])
    short_boom.tag          = 'boom_2l'
    short_boom.index        = 1
    vehicle.append_component(short_boom)

    short_boom              = deepcopy(vehicle.booms.boom_2r)
    short_boom.origin[0][1] = - (short_boom.origin[0][1] + short_boom.y_pitch)
    short_boom.tag          = 'boom_3l'
    short_boom.index        = 1
    vehicle.append_component(short_boom)

    #------------------------------------------------------------------
    # Nacelles
    #------------------------------------------------------------------ 
    rotor_nacelle                         = MARC.Components.Nacelles.Nacelle()
    rotor_nacelle.tag                     = 'rotor_nacelle'    
    rotor_nacelle_origins                 =  [[0.543,  1.63  , -0.126] ,[0.543, -1.63  ,  -0.126 ] ,
                                             [3.843,  1.63  , -0.126] ,[3.843, -1.63  ,  -0.126] ,
                                             [0.543,  2.826 , -0.126] ,[0.543, -2.826 ,  -0.126 ] ,
                                             [3.843,  2.826 , -0.126] ,[3.843, -2.826 ,  -0.126] ,
                                             [0.543,  4.022 , -0.126] ,[0.543, -4.022 ,  -0.126 ] ,
                                             [3.843,  4.022 , -0.126] ,[3.843, -4.022 ,  -0.126 ]]
    rotor_nacelle.length         = 0.25
    rotor_nacelle.diameter       = 0.25
    rotor_nacelle.orientation_euler_angles  = [0,-90*Units.degrees,0.]    
    rotor_nacelle.flow_through   = False  
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_1'
    nac_segment.percent_x_location = 0.0  
    nac_segment.height             = 0.2
    nac_segment.width              = 0.2
    rotor_nacelle.append_segment(nac_segment)    
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_2'
    nac_segment.percent_x_location = 0.25  
    nac_segment.height             = 0.25
    nac_segment.width              = 0.25
    rotor_nacelle.append_segment(nac_segment)    
    
    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_3'
    nac_segment.percent_x_location = 0.5 
    nac_segment.height             = 0.3
    nac_segment.width              = 0.3
    rotor_nacelle.append_segment(nac_segment)    

    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_4'
    nac_segment.percent_x_location = 0.75
    nac_segment.height             = 0.25
    nac_segment.width              = 0.25
    rotor_nacelle.append_segment(nac_segment)        

    nac_segment                    = MARC.Components.Lofted_Body_Segment.Segment()
    nac_segment.tag                = 'segment_5'
    nac_segment.percent_x_location = 1.0
    nac_segment.height             = 0.2
    nac_segment.width              = 0.2
    rotor_nacelle.append_segment(nac_segment)      
    
    rotor_nacelle.areas.wetted            =  np.pi*rotor_nacelle.diameter*rotor_nacelle.length + 0.5*np.pi*rotor_nacelle.diameter**2     

    for idx in range(12):
        nacelle          = deepcopy(rotor_nacelle)
        nacelle.tag      = 'nacelle_' +  str(idx)
        nacelle.origin   = [rotor_nacelle_origins[idx]] 
        vehicle.append_component(nacelle)  
        
    #------------------------------------------------------------------
    # network
    #------------------------------------------------------------------
    net                              = Battery_Electric_Rotor()
    net.rotor_group_indexes          = [0,1,1,1,1,1,1,1,1,1,1,1,1]
    net.motor_group_indexes          = [0,1,1,1,1,1,1,1,1,1,1,1,1]  
    net.esc_group_indexes            = [0,1,1,1,1,1,1,1,1,1,1,1,1]     
    net.active_propulsor_groups      = [True,True]
    net.nacelle_diameter             = 0.6 * Units.feet
    net.engine_length                = 0.5 * Units.feet
    net.areas                        = Data()
    net.areas.wetted                 = np.pi*net.nacelle_diameter*net.engine_length + 0.5*np.pi*net.nacelle_diameter**2
    net.voltage                      = 500.
 

    #------------------------------------------------------------------
    # Design Payload
    #------------------------------------------------------------------
    payload                      = MARC.Components.Energy.Peripherals.Avionics()
    payload.power_draw           = 0.
    payload.mass_properties.mass = 200. * Units.kg
    net.payload                  = payload

    #------------------------------------------------------------------
    # Design Avionics
    #------------------------------------------------------------------
    avionics            = MARC.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 200. * Units.watts
    net.avionics        = avionics

    #------------------------------------------------------------------
    # Design Battery
    #------------------------------------------------------------------
    bat                      = MARC.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion_LiNiMnCoO2_18650() 
    bat.mass_properties.mass = 500. * Units.kg  
    bat.pack.max_voltage     = net.voltage   
    initialize_from_mass(bat)
    
    # Here we, are going to assume a battery pack module shape. This step is optional but
    # required for thermal analysis of tge pack
    number_of_modules                                  = 10
    bat.module.geometrtic_configuration.total          = int(np.ceil(bat.pack.electrical_configuration.total/number_of_modules))
    bat.module.geometrtic_configuration.normal_count   = int(np.ceil(bat.module.geometrtic_configuration.total/bat.pack.electrical_configuration.series))
    bat.module.geometrtic_configuration.parallel_count = int(np.ceil(bat.module.geometrtic_configuration.total/bat.pack.electrical_configuration.parallel))
    net.battery                                        = bat       
    
    # --------------------------------------------------------------
    # Forward Cruise Propulsor System 
    # --------------------------------------------------------------
    # 1. Electronic Speed Controller 
    propeller_esc            = MARC.Components.Energy.Distributors.Electronic_Speed_Controller() 
    propeller_esc.efficiency = 0.95  
    propeller_esc.tag        = 'propeller_esc'  
    net.electronic_speed_controllers.append(propeller_esc) 
    
    # 2. Propeller 
    g                                = 9.81                                   # gravitational acceleration 
    speed_of_sound                   = 340                                    # speed of sound 
    Drag                             = estimate_cruise_drag(vehicle,altitude = 1400. * Units.ft,speed= 110.* Units['mph'] ,lift_coefficient = 0.5 ,profile_drag = 0.06)
    Hover_Load                       = vehicle.mass_properties.takeoff*g      # hover load  
    
    # Thrust Propeller
    propeller                                   = MARC.Components.Energy.Converters.Propeller()
    propeller.number_of_blades                  = 3
    propeller.tip_radius                        = 1.0668
    propeller.hub_radius                        = 0.21336
    propeller.cruise.design_freestream_velocity = 110.* Units['mph']
    propeller.cruise.design_tip_mach            = 0.5
    propeller.cruise.design_angular_velocity    = propeller.cruise.design_tip_mach *speed_of_sound/propeller.tip_radius
    propeller.cruise.design_Cl                  = 0.7
    propeller.cruise.design_altitude            = 1000 * Units.feet
    propeller.cruise.design_thrust              = Drag*3
    propeller.variable_pitch                    = True  
    airfoil                                     = MARC.Components.Airfoils.Airfoil()   
    airfoil.coordinate_file                     = '../Vehicles/Airfoils/NACA_4412.txt'
    airfoil.polar_files                         = ['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ] 
    propeller.append_airfoil(airfoil)       
    propeller.airfoil_polar_stations            = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    propeller                                   = propeller_design(propeller) 
    propeller.origin                            = [[16.*0.3048 , 0. ,0.575]]
    net.rotors.append(propeller)
    

    # 3. Propeller Motors 
    propeller_motor                      = MARC.Components.Energy.Converters.Motor()
    propeller_motor.tag                  = 'propeller_motor'
    propeller_motor.efficiency           = 0.95
    propeller_motor.nominal_voltage      = bat.pack.max_voltage
    propeller_motor.mass_properties.mass = 2.0  * Units.kg
    propeller_motor.origin               = propeller.origin 
    propeller_motor.no_load_current      = 2.0 
    propeller_motor.rotor_radius         = propeller.tip_radius
    propeller_motor.design_torque        = propeller.cruise.design_torque
    propeller_motor.angular_velocity     = propeller.cruise.design_angular_velocity/propeller_motor.gear_ratio     
    propeller_motor                      = size_optimal_motor(propeller_motor) 
    net.motors.append(propeller_motor)
     
    
    # --------------------------------------------------------------
    # Lift Propulsor System 
    # -------------------------------------------------------------- 
    # 1. Electronic Speed Controller 
    lift_rotor_esc              = MARC.Components.Energy.Distributors.Electronic_Speed_Controller() 
    lift_rotor_esc.efficiency   = 0.95 
    for ii in range(12):
        lift_rotor_ESC          = deepcopy(lift_rotor_esc)
        lift_rotor_ESC.tag      = 'lift_rotor_esc' + str(ii + 1)  
        net.electronic_speed_controllers.append(lift_rotor_ESC)  
    
    # 2. Lift Rotors  
    lift_rotor                                  = MARC.Components.Energy.Converters.Propeller() # using propeller for for regression! 
    lift_rotor.tip_radius                       = 2.8 * Units.feet
    lift_rotor.hub_radius                       = 0.35 * Units.feet
    lift_rotor.number_of_blades                 = 2 
    lift_rotor.orientation_euler_angles         = [0.,np.pi/2.,0.] # This is Z-direction thrust up in vehicle frame    
    lift_rotor.cruise.design_freestream_velocity= 500. * Units['ft/min']
    lift_rotor.cruise.design_tip_mach           = 0.65
    lift_rotor.cruise.design_angular_velocity   = lift_rotor.cruise.design_tip_mach* speed_of_sound /lift_rotor.tip_radius
    lift_rotor.cruise.design_Cl                 = 0.7
    lift_rotor.cruise.design_altitude           = 20 * Units.feet
    lift_rotor.cruise.design_thrust             = Hover_Load/(12-1) # contingency for one-engine-inoperative condition
    lift_rotor.variable_pitch                   = True  
    airfoil                                     = MARC.Components.Airfoils.Airfoil()   
    airfoil.coordinate_file                     = '../Vehicles/Airfoils/NACA_4412.txt'
    airfoil.polar_files                         = ['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                                '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ] 
    lift_rotor.append_airfoil(airfoil)          
    lift_rotor.airfoil_polar_stations           = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    lift_rotor                                  = propeller_design(lift_rotor)  
   
    # Appending rotors with different origins
    rotations = [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]
    origins   = [[0.543,  1.63  , -0.126] ,[0.543, -1.63  ,  -0.126],
                 [3.843,  1.63  , -0.126] ,[3.843, -1.63  ,  -0.126],
                 [0.543,  2.826 , -0.126] ,[0.543, -2.826 ,  -0.126],
                 [3.843,  2.826 , -0.126] ,[3.843, -2.826 ,  -0.126],
                 [0.543,  4.022 , -0.126] ,[0.543, -4.022 ,  -0.126],
                 [3.843,  4.022 , -0.126] ,[3.843, -4.022 ,  -0.126]]

    for ii in range(12):
        lift_rotor          = deepcopy(lift_rotor)
        lift_rotor.tag      = 'lift_rotor'
        lift_rotor.rotation = rotations[ii]
        lift_rotor.origin   = [origins[ii]]
        net.rotors.append(lift_rotor) 
         
    # 3. Lift Rotor Motors 
    lift_rotor_motor                          = MARC.Components.Energy.Converters.Motor()
    lift_rotor_motor.efficiency               = 0.85
    lift_rotor_motor.nominal_voltage          = bat.pack.max_voltage*3/4 
    lift_rotor_motor.mass_properties.mass     = 3. * Units.kg
    lift_rotor_motor.origin                   = lift_rotor.origin
    lift_rotor_motor.gearbox_efficiency       = 1.0
    lift_rotor_motor.no_load_current          = 4.0 
    lift_rotor_motor.rotor_radius             = lift_rotor.tip_radius
    lift_rotor_motor.design_torque            = lift_rotor.cruise.design_torque
    lift_rotor_motor.angular_velocity         = lift_rotor.cruise.design_angular_velocity/lift_rotor_motor.gear_ratio   
    lift_rotor_motor                          = size_optimal_motor(lift_rotor_motor)

    # Appending motors with different origins
    for _ in range(12):
        lift_rotor_motor = deepcopy(lift_rotor_motor)
        lift_rotor_motor.tag = 'lift_motor'
        net.motors.append(lift_rotor_motor)


    # append motor origin spanwise locations onto wing data structure
    vehicle.append_component(net)

    # Add extra drag sources from motors, props, and landing gear. All of these hand measured
    motor_height                     = .25 * Units.feet
    motor_width                      = 1.6 * Units.feet
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
    total_excrescence_area_spin      = 12.*motor_height*motor_width + 2.*main_gear_length*main_gear_width \
        + nose_gear_width*nose_gear_length + 2*main_tire_height*main_tire_width\
        + nose_tire_height*nose_tire_width
    total_excrescence_area_no_spin   = total_excrescence_area_spin + 12*propeller_height*propeller_width
    vehicle.excrescence_area_no_spin = total_excrescence_area_no_spin
    vehicle.excrescence_area_spin    = total_excrescence_area_spin

    vehicle.wings['main_wing'].motor_spanwise_locations = np.multiply(
        2./36.25,
        [-5.435, -5.435, -9.891, -9.891, -14.157, -14.157,
         5.435, 5.435, 9.891, 9.891, 14.157, 14.157])

    vehicle.wings['main_wing'].winglet_fraction        = 0.0
    vehicle.wings['main_wing'].thickness_to_chord      = 0.18
    vehicle.wings['main_wing'].chords.mean_aerodynamic = 0.9644599977664836

    vehicle.weight_breakdown  = empty(vehicle,settings=Data())
    compute_component_centers_of_gravity(vehicle)
    vehicle.center_of_gravity()

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
    base_config.networks.battery_electric_rotor.active_propulsor_groups= [True,True]
    configs.append(base_config)


    forward_config = MARC.Components.Configs.Config(vehicle)
    forward_config.tag = 'forward_flight'
    forward_config.networks.battery_electric_rotor.pitch_command = 0
    forward_config.networks.battery_electric_rotor.active_propulsor_groups = [True,False]
    configs.append(forward_config) 


    transition_config = MARC.Components.Configs.Config(vehicle)
    transition_config.tag = 'transition_flight'
    transition_config.networks.battery_electric_rotor.pitch_command = 0
    transition_config.networks.battery_electric_rotor.active_propulsor_groups = [True,True]
    configs.append(transition_config)
    

    vertical_config = MARC.Components.Configs.Config(vehicle)
    vertical_config.tag = 'vertical_flight' 
    vertical_config.networks.battery_electric_rotor.pitch_command = 0
    vertical_config.networks.battery_electric_rotor.active_propulsor_groups = [False,True]
    configs.append(vertical_config)    
 
    # done!
    return configs

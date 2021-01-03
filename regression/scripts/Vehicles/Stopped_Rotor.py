# Stopped_Rotor_CRM.py
# 
# Created: May 2019, M Clarke
#          Sep 2020, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data 
import copy
from SUAVE.Components.Energy.Networks.Lift_Cruise              import Lift_Cruise
from SUAVE.Methods.Power.Battery.Sizing                        import initialize_from_mass
from SUAVE.Methods.Weights.Correlations.Propulsion             import nasa_motor
from SUAVE.Methods.Propulsion.electric_motor_sizing            import size_optimal_motor
from SUAVE.Methods.Propulsion                                  import propeller_design   
from SUAVE.Methods.Weights.Buildups.Electric_Lift_Cruise.empty import empty
from SUAVE.Plots.Geometry_Plots import * 

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
    vehicle               = SUAVE.Vehicle()
    vehicle.tag           = 'Stopped_Rotor'
    vehicle.configuration = 'eVTOL'

    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff           = 2086 * Units.lb 
    vehicle.mass_properties.operating_empty   = 2086. * Units.lb               # Approximate
    vehicle.mass_properties.max_takeoff       = 2086. * Units.lb               # Approximate
    vehicle.mass_properties.center_of_gravity = [[2.0144,   0.  ,  0. ]] # Approximate

    # basic parameters                          
    vehicle.reference_area         = 10.76  
    vehicle.envelope.ultimate_load = 5.7   
    vehicle.envelope.limit_load    = 3.  

    # ------------------------------------------------------------------    
    # WINGS                                    
    # ------------------------------------------------------------------    
    # WING PROPERTIES           
    wing                          = SUAVE.Components.Wings.Main_Wing()
    wing.tag                      = 'main_wing'  
    wing.aspect_ratio             = 12.000
    wing.sweeps.quarter_chord     = 0.0  * Units.degrees
    wing.thickness_to_chord       = 0.18  
    wing.taper                    = 1.  
    wing.spans.projected          = 36.0   * Units.feet
    wing.chords.root              = 6.5    * Units.feet
    wing.total_length             = 6.5    * Units.feet 
    wing.chords.tip               = 3.     * Units.feet 
    wing.chords.mean_aerodynamic  = 3.     * Units.feet  
    wing.dihedral                 = 1.0    * Units.degrees  
    wing.areas.reference          = 10.582 * Units.meter**2 
    wing.areas.wetted             = 227.5  * Units.feet**2  
    wing.areas.exposed            = 227.5  * Units.feet**2  
    wing.twists.root              = 0.0    * Units.degrees  
    wing.twists.tip               = 0.0    * Units.degrees   
    wing.origin                   = [[  1.067, 0., -0.261 ]]
    wing.aerodynamic_center       = [1.975 , 0., -0.261]    
    wing.winglet_fraction         = 0.0  
    wing.symmetric                = True
    wing.vertical                 = False

    # Segment                                  
    segment                       = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'Section_1'   
    segment.percent_span_location = 0.  
    segment.twist                 = 0.  
    segment.root_chord_percent    = 1. 
    segment.dihedral_outboard     = 2.00 * Units.degrees
    segment.sweeps.quarter_chord  = 30.00 * Units.degrees 
    segment.thickness_to_chord    = 0.18  
    wing.Segments.append(segment)               

    # Segment                                   
    segment                       = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'Section_2'    
    segment.percent_span_location = 0.141 
    segment.twist                 = 0. 
    segment.root_chord_percent    = 0.461383 
    segment.dihedral_outboard     = 2.00 * Units.degrees
    segment.sweeps.quarter_chord  = 0. 
    segment.thickness_to_chord    = 0.16 
    wing.Segments.append(segment)               

    # Segment                                   
    segment                       = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'Section_3'   
    segment.percent_span_location = 0.8478 
    segment.twist                 = 0. 
    segment.root_chord_percent    = 0.461383   
    segment.dihedral_outboard     = 10.00 * Units.degrees 
    segment.sweeps.quarter_chord  = 17.00 * Units.degrees 
    segment.thickness_to_chord    = 0.16 
    wing.Segments.append(segment)               

    # Segment                                  
    segment                       = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'Section_4'   
    segment.percent_span_location = 0.96726 
    segment.twist                 = 0. 
    segment.root_chord_percent    = 0.323 
    segment.dihedral_outboard     = 20.0    * Units.degrees 
    segment.sweeps.quarter_chord  = 51.000  * Units.degrees 
    segment.thickness_to_chord    = 0.16  
    wing.Segments.append(segment)                
    
    # Segment                                   
    segment                       = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'Section_6'   
    segment.percent_span_location = 1.0 
    segment.twist                 = 0.  
    segment.root_chord_percent    = 0.0413890
    segment.dihedral_outboard     = 1.0  * Units.degrees
    segment.sweeps.quarter_chord  = 0.0 * Units.degrees
    segment.thickness_to_chord    = 0.16 
    wing.Segments.append(segment)   

    # add to vehicle
    vehicle.append_component(wing)       

    # WING PROPERTIES 
    wing                          = SUAVE.Components.Wings.Wing()
    wing.tag                      = 'horizontal_tail'  
    wing.aspect_ratio             = 4.78052
    wing.sweeps.quarter_chord     = 0.0  
    wing.thickness_to_chord       = 0.12  
    wing.taper                    = 1.0  
    wing.spans.projected          = 2.914 
    wing.chords.root              = 0.609
    wing.total_length             = 0.609
    wing.chords.tip               = 0.609
    wing.chords.mean_aerodynamic  = 0.609
    wing.dihedral                 = 0.  * Units.degrees  
    wing.areas.reference          = 5.82204
    wing.areas.wetted             = 5.82204*2 * Units.feet**2    
    wing.areas.exposed            = 5.82204*2 * Units.feet**2  
    wing.twists.root              = 0. * Units.degrees  
    wing.twists.tip               = 0. * Units.degrees  
    wing.origin                   = [[5.440, 0.0 , 1.28]]
    wing.aerodynamic_center       = [5.7,  0.,  0.] 
    wing.winglet_fraction         = 0.0 
    wing.symmetric                = True    

    # add to vehicle
    vehicle.append_component(wing)    


    # WING PROPERTIES
    wing                          = SUAVE.Components.Wings.Wing()
    wing.tag                      = 'vertical_tail_1'
    wing.aspect_ratio             = 4.30556416
    wing.sweeps.quarter_chord     = 13.68 * Units.degrees 
    wing.thickness_to_chord       = 0.12
    wing.taper                    = 0.5 
    wing.spans.projected          = 1.6  
    wing.chords.root              = 1.2192
    wing.total_length             = 1.2192
    wing.chords.tip               = 0.6096
    wing.chords.mean_aerodynamic  = 0.9144
    wing.areas.reference          = 2.357
    wing.areas.wetted             = 2.357*2 * Units.feet**2 
    wing.areas.exposed            = 2.357*2 * Units.feet**2 
    wing.twists.root              = 0. * Units.degrees 
    wing.twists.tip               = 0. * Units.degrees  
    wing.origin                   = [[4.900 ,  -1.657 ,  -0.320 ]]
    wing.aerodynamic_center       = 0.0   
    wing.winglet_fraction         = 0.0  
    wing.dihedral                 = 6.  * Units.degrees  
    wing.vertical                 = True 
    wing.symmetric                = False

    # add to vehicle
    vehicle.append_component(wing)   


    # WING PROPERTIES
    wing                         = SUAVE.Components.Wings.Wing()
    wing.tag                     = 'vertical_tail_2'
    wing.aspect_ratio            = 4.30556416
    wing.sweeps.quarter_chord    = 13.68 * Units.degrees 
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 0.5 
    wing.spans.projected         = 1.6  
    wing.chords.root             = 1.2192
    wing.total_length            = 1.2192
    wing.chords.tip              = 0.6096
    wing.chords.mean_aerodynamic = 0.8
    wing.areas.reference         = 2.357
    wing.areas.wetted            = 2.357*2 * Units.feet**2 
    wing.areas.exposed           = 2.357*2 * Units.feet**2 
    wing.twists.root             = 0. * Units.degrees 
    wing.twists.tip              = 0. * Units.degrees  
    wing.origin                  = [[4.900 ,  1.657 ,  -0.320 ]]
    wing.aerodynamic_center      = 0.0   
    wing.winglet_fraction        = 0.0  
    wing.dihedral                = -6.  * Units.degrees  
    wing.vertical                = True   
    wing.symmetric               = False

    # add to vehicle
    vehicle.append_component(wing)   

    # ---------------------------------------------------------------   
    # FUSELAGE                
    # ---------------------------------------------------------------   
    # FUSELAGE PROPERTIES
    fuselage                                    = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage'
    fuselage.configuration                      = 'Tube_Wing'  
    fuselage.seats_abreast                      = 2.  
    fuselage.seat_pitch                         = 3.  
    fuselage.fineness.nose                      = 0.88   
    fuselage.fineness.tail                      = 1.13   
    fuselage.lengths.nose                       = 3.2 * Units.feet 
    fuselage.lengths.tail                       = 6.4 * Units.feet
    fuselage.lengths.cabin                      = 6.4 * Units.feet 
    fuselage.lengths.total                      = 4.10534
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
    segment                           = SUAVE.Components.Fuselages.Segment() 
    segment.tag                       = 'segment_0'   
    segment.percent_x_location        = 0.
    segment.percent_z_location        = -0.267/4.10534
    segment.height                    = 0.1  
    segment.width                     = 0.1  
    fuselage.Segments.append(segment)           

    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_1'   
    segment.percent_x_location        =  0.2579 /4.10534
    segment.percent_z_location        = -0.12881/4.10534
    segment.height                    = 0.5201 
    segment.width                     = 0.75  
    fuselage.Segments.append(segment) 

    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_2'   
    segment.percent_x_location        =  0.9939/4.10534
    segment.percent_z_location        =  -0.0446/4.10534
    segment.height                    =  1.18940
    segment.width                     =  1.42045
    fuselage.Segments.append(segment)           

    # Segment                                  
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_3'   
    segment.percent_x_location        =   1.95060 /4.10534
    segment.percent_z_location        =  0/4.10534
    segment.height                    =  1.37248
    segment.width                     =  1.35312
    fuselage.Segments.append(segment)           

    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_4'   
    segment.percent_x_location        = 3.02797/4.10534
    segment.percent_z_location        = 0.08/4.10534
    segment.height                    = 0.8379
    segment.width                     = 0.79210 
    fuselage.Segments.append(segment)    
    
    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_5'   
    segment.percent_x_location        =  1.
    segment.percent_z_location        =  0.12522/4.10534
    segment.height                    = 0.05
    segment.width                     = 0.05 
    fuselage.Segments.append(segment)           
    

    # add to vehicle
    vehicle.append_component(fuselage) 

    #-------------------------------------------------------------------
    # INNER BOOMS   
    #-------------------------------------------------------------------   
    long_boom                                    = SUAVE.Components.Fuselages.Fuselage()
    long_boom.tag                                = 'boom_1r'
    long_boom.configuration                      = 'boom'  
    long_boom.origin                             = [[0.543,1.630, -0.326]]  
    long_boom.seats_abreast                      = 0.  
    long_boom.seat_pitch                         = 0.0 
    long_boom.fineness.nose                      = 0.950   
    long_boom.fineness.tail                      = 1.029   
    long_boom.lengths.nose                       = 0.2 
    long_boom.lengths.tail                       = 0.2
    long_boom.lengths.cabin                      = 5.2 
    long_boom.lengths.total                      = 5.6
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
    segment                           = SUAVE.Components.Fuselages.Segment() 
    segment.tag                       = 'segment_1'   
    segment.percent_x_location        = 0.
    segment.percent_z_location        = 0.0 
    segment.height                    = 0.05  
    segment.width                     = 0.05   
    long_boom.Segments.append(segment)           

    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_2'   
    segment.percent_x_location        = 0.2/ 5.6
    segment.percent_z_location        = 0. 
    segment.height                    = 0.15 
    segment.width                     = 0.15 
    long_boom.Segments.append(segment) 

    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_3'   
    segment.percent_x_location        = 5.4/5.6 
    segment.percent_z_location        = 0. 
    segment.height                    = 0.15
    segment.width                     = 0.15
    long_boom.Segments.append(segment)           

    # Segment                                  
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_4'    
    segment.percent_x_location        = 1.   
    segment.percent_z_location        = 0.   
    segment.height                    = 0.05   
    segment.width                     = 0.05   
    long_boom.Segments.append(segment)           
    
    # add to vehicle
    vehicle.append_component(long_boom)   
    
    # add left long boom 
    long_boom              = deepcopy(vehicle.fuselages.boom_1r)
    long_boom.origin[0][1] = -long_boom.origin[0][1]
    long_boom.tag          = 'Boom_1L'
    long_boom.index        = 1 
    vehicle.append_component(long_boom) 
            
            
    #-------------------------------------------------------------------
    # OUTER BOOMS   
    #-------------------------------------------------------------------   
    short_boom                                    = SUAVE.Components.Fuselages.Fuselage()
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
    short_boom.symmetric                          = True 
    short_boom.index                              = 1

    # Segment  
    segment                           = SUAVE.Components.Fuselages.Segment() 
    segment.tag                       = 'segment_1'   
    segment.percent_x_location        = 0.
    segment.percent_z_location        = 0.0 
    segment.height                    = 0.05  
    segment.width                     = 0.05   
    short_boom.Segments.append(segment)           

    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_2'   
    segment.percent_x_location        =  0.2/3.3
    segment.percent_z_location        = 0. 
    segment.height                    = 0.15 
    segment.width                     = 0.15 
    short_boom.Segments.append(segment) 

    # Segment                                   
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_3'   
    segment.percent_x_location        = 3.1/3.3
    segment.percent_z_location        = 0. 
    segment.height                    = 0.15
    segment.width                     = 0.15
    short_boom.Segments.append(segment)           

    # Segment                                  
    segment                           = SUAVE.Components.Fuselages.Segment()
    segment.tag                       = 'segment_4'    
    segment.percent_x_location        = 1.   
    segment.percent_z_location        = 0.   
    segment.height                    = 0.05   
    segment.width                     = 0.05   
    short_boom.Segments.append(segment)       

    # add to vehicle
    vehicle.append_component(short_boom)    
    
    # add outer right boom 
    short_boom              = deepcopy(vehicle.fuselages.boom_2r)
    short_boom.origin[0][1] = 1.196  + short_boom.origin[0][1]
    short_boom.tag          = 'boom_3r'
    short_boom.index        = 1 
    vehicle.append_component(short_boom)  
    
    # add inner left boom 
    short_boom              = deepcopy(vehicle.fuselages.boom_2r)
    short_boom.origin[0][1] = - (short_boom.origin[0][1])
    short_boom.tag          = 'boom_2l'
    short_boom.index        = 1 
    vehicle.append_component(short_boom)     
    
    short_boom              = deepcopy(vehicle.fuselages.boom_2r)
    short_boom.origin[0][1] = - (short_boom.origin[0][1] + 1.196 )
    short_boom.tag          = 'boom_3l'
    short_boom.index        = 1 
    vehicle.append_component(short_boom) 

    #------------------------------------------------------------------
    # PROPULSOR
    #------------------------------------------------------------------
    net                             = Lift_Cruise()
    net.number_of_rotor_engines     = 12
    net.number_of_propeller_engines = 1
    net.rotor_thrust_angle          = 90. * Units.degrees
    net.propeller_thrust_angle      = 0. 
    net.rotor_nacelle_diameter      = 0.6 * Units.feet  
    net.rotor_engine_length         = 0.5 * Units.feet
    net.propeller_nacelle_diameter  = 0.6 * Units.feet  
    net.propeller_engine_length     = 0.5 * Units.feet    
    net.areas                       = Data()
    net.areas.wetted                = np.pi*net.nacelle_diameter*net.engine_length + 0.5*np.pi*net.nacelle_diameter**2    
    net.voltage                     = 400.

    #------------------------------------------------------------------
    # Design Electronic Speed Controller 
    #------------------------------------------------------------------
    rotor_esc              = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    rotor_esc.efficiency   = 0.95
    net.rotor_esc          = rotor_esc 

    propeller_esc            = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    propeller_esc.efficiency = 0.95
    net.propeller_esc        = propeller_esc

    #------------------------------------------------------------------
    # Design Payload
    #------------------------------------------------------------------
    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 10. # Watts 
    payload.mass_properties.mass = 200.0 * Units.kg
    net.payload                  = payload

    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 20. # Watts  
    net.avionics        = avionics    

    #------------------------------------------------------------------
    # Design Battery
    #------------------------------------------------------------------ 
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 500. * Units.kg  
    bat.specific_energy      = 200. * Units.Wh/Units.kg
    bat.resistance           = 0.006
    bat.max_voltage          = net.voltage
    
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat 
    net.voltage              = bat.max_voltage
    
    # Component 9 Miscellaneous Systems 
    sys = SUAVE.Components.Systems.System()
    sys.mass_properties.mass = 5 # kg

    #------------------------------------------------------------------
    # Design Rotors and Propellers
    #------------------------------------------------------------------
    # atmosphere and flight conditions for propeller/rotor design
    g              = 9.81                                   # gravitational acceleration 
    S              = vehicle.reference_area                 # reference area 
    speed_of_sound = 340                                    # speed of sound 
    rho            = 1.22                                   # reference density
    fligth_CL      = 0.75                                   # cruise target lift coefficient 
    AR             = vehicle.wings.main_wing.aspect_ratio   # aspect ratio 
    Cd0            = 0.06                                   # profile drag
    Cdi            = fligth_CL**2/(np.pi*AR*0.98)           # induced drag
    Cd             = Cd0 + Cdi                              # total drag
    V_inf          = 110.* Units['mph']                     # freestream velocity 
    Drag           = S * (0.5*rho*V_inf**2 )*Cd             # cruise drag
    Hover_Load     = vehicle.mass_properties.takeoff*g      # hover load  

    # Thrust Propeller                          
    propeller                     = SUAVE.Components.Energy.Converters.Propeller() 
    propeller.number_of_blades         = 3
    propeller.number_of_engines   = net.number_of_propeller_engines
    propeller.freestream_velocity = V_inf
    propeller.tip_radius          = 1.0668
    propeller.hub_radius          = 0.21336 
    propeller.design_tip_mach     = 0.5  
    propeller.angular_velocity    = propeller.design_tip_mach *speed_of_sound/propeller.tip_radius   
    propeller.design_Cl           = 0.7
    propeller.design_altitude     = 1000 * Units.feet   
    propeller.design_thrust       = (Drag*2.5)/net.number_of_propeller_engines 
    propeller                     = propeller_design(propeller)   
    propeller.origin              = [[4.10534,0,0.12522]]   
    net.propeller                 = propeller
 
    # Lift Rotors                               
    rotor                         = SUAVE.Components.Energy.Converters.Rotor() 
    rotor.tip_radius              = 2.8 * Units.feet
    rotor.hub_radius              = 0.35 * Units.feet      
    rotor.number_of_blades        = 2
    rotor.design_tip_mach         = 0.65 
    rotor.number_of_engines       = net.number_of_rotor_engines
    rotor.disc_area               = np.pi*(rotor.tip_radius**2)        
    rotor.induced_hover_velocity  = np.sqrt(Hover_Load/(2*rho*rotor.disc_area*net.number_of_rotor_engines)) 
    rotor.freestream_velocity     = 500. * Units['ft/min']  
    rotor.angular_velocity        = rotor.design_tip_mach* speed_of_sound /rotor.tip_radius   
    rotor.design_Cl               = 0.7
    rotor.design_altitude         = 20 * Units.feet                            
    rotor.design_thrust           = (Hover_Load* 2.5)/net.number_of_rotor_engines   
    rotor                         = propeller_design(rotor)          
    rotor.origin                  = vehicle.fuselages['boom_1r'].origin
    rotor.symmetric               = True  
    rotor.rotation                = [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]
    rotor.origin                  = [[0.543,  1.63  , -0.126] ,[0.543, -1.63  ,  -0.126 ] , 
                                     [3.843,  1.63  , -0.126] ,[3.843, -1.63  ,  -0.126] ,
                                     [0.543,  2.826 , -0.126] ,[0.543, -2.826 ,  -0.126 ] , 
                                     [3.843,  2.826 , -0.126] ,[3.843, -2.826 ,  -0.126] , 
                                     [0.543,  4.022 , -0.126] ,[0.543, -4.022 ,  -0.126 ] , 
                                     [3.843,  4.022 , -0.126] ,[3.843, -4.022 ,  -0.126 ]]
    rotor_motor_origins           = np.array(rotor.origin)  
    net.rotor = rotor

    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    # Propeller (Thrust) motor
    propeller_motor                      = SUAVE.Components.Energy.Converters.Motor()
    propeller_motor.efficiency           = 0.95
    propeller_motor.nominal_voltage      = bat.max_voltage  
    propeller_motor.origin               = propeller.origin  
    propeller_motor.propeller_radius     = propeller.tip_radius      
    propeller_motor.no_load_current      = 2.0  
    propeller_motor                      = size_optimal_motor(propeller_motor,propeller)
    propeller_motor.mass_properties.mass = nasa_motor(propeller_motor.design_torque)
    net.propeller_motor                  = propeller_motor

    # Rotor (Lift) Motor                        
    rotor_motor                         = SUAVE.Components.Energy.Converters.Motor()
    rotor_motor.efficiency              = 0.95  
    rotor_motor.nominal_voltage         = bat.max_voltage  
    rotor_motor.origin                  = rotor.origin  
    rotor_motor.propeller_radius        = rotor.tip_radius   
    rotor_motor.gearbox_efficiency      = 1.0 
    rotor_motor.no_load_current         = 4.0   
    rotor_motor                         = size_optimal_motor(rotor_motor,rotor)
    rotor_motor.mass_properties.mass    = nasa_motor(rotor_motor.design_torque)
    net.rotor_motor                     = rotor_motor   
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

    vehicle.wings['main_wing'].motor_spanwise_locations = np.multiply(2./wing.spans.projected ,rotor_motor_origins[:,1])  
    vehicle.wings['horizontal_tail'].motor_spanwise_locations = np.array([0.])
    vehicle.wings['vertical_tail_1'].motor_spanwise_locations = np.array([0.])
    vehicle.wings['vertical_tail_2'].motor_spanwise_locations = np.array([0.]) 
    
    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------
    
    # plot vehicle 
    plot_vehicle(vehicle,plot_control_points = False) 
    
    return vehicle
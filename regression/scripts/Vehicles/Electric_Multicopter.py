# Electric_Multicopter.py
# 
# Created: Feb 2020, M Clarke
#          Sep 2020, M. Clarke 

#----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data 
from SUAVE.Components.Energy.Networks.Vectored_Thrust import Vectored_Thrust
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass 
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Aerodynamics.Fidelity_Zero.Lift import compute_max_lift_coeff 
from SUAVE.Methods.Weights.Buildups.eVTOL.empty import empty 
from SUAVE.Methods.Propulsion.electric_motor_sizing            import size_from_mass , size_optimal_motor
from SUAVE.Methods.Weights.Correlations.Propulsion import nasa_motor, hts_motor , air_cooled_motor
import numpy as np

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup(): 
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    vehicle                                     = SUAVE.Vehicle()
    vehicle.tag                                 = 'multicopter'
    vehicle.configuration                       = 'eVTOL'
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff             = 2080. * Units.lb 
    vehicle.mass_properties.operating_empty     = 1666. * Units.lb            
    vehicle.mass_properties.max_takeoff         = 2080. * Units.lb               
    vehicle.mass_properties.center_of_gravity   = [[2.6, 0., 0. ] ] 
                                                
    # This needs updating                       
    vehicle.passengers                          = 5
    vehicle.reference_area                      = 2. 
    vehicle.envelope.ultimate_load              = 5.7   
    vehicle.envelope.limit_load                 = 3.  
                                                
    wing = SUAVE.Components.Wings.Main_Wing()  # this is the body of the vehicle 
    wing.tag                      = 'main_wing'   
    wing.aspect_ratio             = 0.5 
    wing.sweeps.quarter_chord     = 0.  
    wing.thickness_to_chord       = 0.1  
    wing.taper                    = 1.  
    wing.spans.projected          = 1. 
    wing.chords.root              = 2.  
    wing.total_length             = 2.  
    wing.chords.tip               = 2.  
    wing.chords.mean_aerodynamic  = 1.  
    wing.dihedral                 = 0.0  
    wing.areas.reference          = 1.   
    wing.areas.wetted             = 1.  
    wing.areas.exposed            = 1.  
    wing.twists.root              = 0.  
    wing.twists.tip               = 0.  
    wing.origin                   = [[1, 0.0 ,0.2]] 
    wing.aerodynamic_center       = [0., 0., 0.]     
    wing.winglet_fraction         = 0.0  
    wing.symmetric                = True 
    
    vehicle.append_component(wing)
    
    # ------------------------------------------------------    
    # FUSELAGE    
    # ------------------------------------------------------    
    # FUSELAGE PROPERTIES
    fuselage                                    = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                                = 'fuselage'
    fuselage.configuration                      = 'Tube_Wing'  
    fuselage.seats_abreast                      = 2.  
    fuselage.seat_pitch                         = 2.  
    fuselage.fineness.nose                      = 0.88   
    fuselage.fineness.tail                      = 1.13   
    fuselage.lengths.nose                       = 0.5 
    fuselage.lengths.tail                       = 0.5
    fuselage.lengths.cabin                      = 3.5 
    fuselage.lengths.total                      = 4.000 
    fuselage.width                              = 1.300
    fuselage.heights.maximum                    = 1.40
    fuselage.heights.at_quarter_length          = 1.40 
    fuselage.heights.at_wing_root_quarter_chord = 1.40
    fuselage.heights.at_three_quarters_length   = 1.40
    fuselage.areas.wetted                       = 19.829265
    fuselage.areas.front_projected              = 1.4294246 
    fuselage.effective_diameter                 = 1.300
    fuselage.differential_pressure              = 1. 
    
    # Segment  
    segment                          = SUAVE.Components.Fuselages.Segment() 
    segment.tag                      = 'segment_0'   
    segment.percent_x_location       = 0.  
    segment.percent_z_location       = 0.0 
    segment.height                   = 0.1   
    segment.width                    = 0.1   
    fuselage.append_segment(segment)            
                                                
    # Segment                                   
    segment                         = SUAVE.Components.Fuselages.Segment()
    segment.tag                     = 'segment_1'   
    segment.percent_x_location      = 0.200/4.
    segment.percent_z_location      = 0.1713/4.
    segment.height                  = 0.737
    segment.width                   = 0.9629
    segment.vsp_data.top_angle      = 53.79 * Units.degrees 
    segment.vsp_data.bottom_angle   = 28.28 * Units.degrees     
    fuselage.append_segment(segment)            
                                                
    # Segment                                   
    segment                         = SUAVE.Components.Fuselages.Segment()
    segment.tag                     = 'segment_2'   
    segment.percent_x_location      = 0.8251/4.
    segment.percent_z_location      = 0.2840/4.
    segment.height                  = 1.40 
    segment.width                   = 1.30 
    segment.vsp_data.top_angle      = 0 * Units.degrees 
    segment.vsp_data.bottom_angle   = 0 * Units.degrees     
    fuselage.append_segment(segment)            
                                                
    # Segment                                  
    segment                         = SUAVE.Components.Fuselages.Segment()
    segment.tag                     = 'segment_3'   
    segment.percent_x_location      = 3.342/4.
    segment.percent_z_location      = 0.356/4.
    segment.height                  = 1.40
    segment.width                   = 1.300
    #segment.vsp_data.top_angle      = 0 * Units.degrees 
    #segment.vsp_data.bottom_angle   = 0 * Units.degrees     
    fuselage.append_segment(segment)  
                                                
    # Segment                                   
    segment                         = SUAVE.Components.Fuselages.Segment()
    segment.tag                     = 'segment_4'   
    segment.percent_x_location      = 3.70004/4.
    segment.percent_z_location      = 0.4636/4.
    segment.height                  = 0.9444
    segment.width                   = 0.9946 
    segment.vsp_data.top_angle      = -36.59 * Units.degrees 
    segment.vsp_data.bottom_angle   = -57.94 * Units.degrees 
    fuselage.append_segment(segment)             
 
    # Segment                                   
    segment                         = SUAVE.Components.Fuselages.Segment()
    segment.tag                     = 'segment_5'   
    segment.percent_x_location      = 1.
    segment.percent_z_location      = 0.6320/4.
    segment.height                  = 0.1    
    segment.width                   = 0.1    
    fuselage.append_segment(segment)             
 
                                                 
    # add to vehicle
    vehicle.append_component(fuselage)   
       
    #------------------------------------------------------------------
    # PROPULSOR
    #------------------------------------------------------------------
    net                     = Vectored_Thrust()
    net.number_of_engines   = 4
    net.thrust_angle        = 90. * Units.degrees
    net.nacelle_diameter    = 1.42*2  
    net.nacelle_start       = 0.5
    net.nacelle_end         = 0.7
    net.nacelle_offset      = 0.0
    net.engine_length       = 0.5
    net.areas               = Data()
    net.areas.wetted        = np.pi*net.nacelle_diameter*net.engine_length + 0.5*np.pi*net.nacelle_diameter**2    
    net.voltage             = 400.
 
    # Component 1:  Electronic Speed Controller  
    esc             = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency  = 0.95
    net.esc         = esc
       
    # Component 2: Rotor
    g               = 9.81                                   # gravitational acceleration  
    speed_of_sound  = 340                                    # speed of sound 
    rho             = 1.22                                   # reference density
    Hover_Load      = vehicle.mass_properties.takeoff*g      # hover load   
    design_tip_mach = 0.8                                    # design tip mach number 
    
    rotor                        = SUAVE.Components.Energy.Converters.Rotor() 
    rotor.tip_radius             = 1.4
    rotor.hub_radius             = 0.1
    rotor.disc_area              = np.pi*(rotor.tip_radius**2) 
    rotor.number_of_blades       = 3
    rotor.freestream_velocity    = 10 # 500. * Units['ft/min']  
    rotor.angular_velocity       = (design_tip_mach*speed_of_sound)/rotor.tip_radius   
    rotor.design_Cl              = 0.8
    rotor.design_altitude        = 1000 * Units.feet                   
    rotor.design_thrust          = (Hover_Load/net.number_of_engines) 
    rotor.airfoil_geometry       =  ['../Vehicles/NACA_4412.txt'] 
    rotor.airfoil_polars         = [['../Vehicles/NACA_4412_polar_Re_50000.txt' ,'../Vehicles/NACA_4412_polar_Re_100000.txt' ,'../Vehicles/NACA_4412_polar_Re_200000.txt' ,
                                     '../Vehicles/NACA_4412_polar_Re_500000.txt' ,'../Vehicles/NACA_4412_polar_Re_1000000.txt' ]]
    rotor.airfoil_polar_stations = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]      
    rotor                        = propeller_design(rotor)     
    rotor.induced_hover_velocity = np.sqrt(Hover_Load/(2*rho*rotor.disc_area*net.number_of_engines))   
    rotor.rotation               = [-1, 1,-1,1]
    rotor.origin                 = [[ 0.870,2.283,1.196],[ 0.870,-2.283,1.196],[4.348,2.283,1.196] ,[4.348,-2.283,1.196]] 
    net.origin                   = [[ 0.870,2.283,1.196],[ 0.870,-2.283,1.196],[4.348,2.283,1.196] ,[4.348,-2.283,1.196]] 
    net.rotor = rotor  
    
    # Component 3: Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 800. * Units.kg  
    bat.specific_energy      = 200. * Units.Wh/Units.kg
    bat.resistance           = 0.006
    bat.max_voltage          = net.voltage
    
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat 
    net.voltage              = bat.max_voltage 
  
    # Component 4: Motor
    motor                        = SUAVE.Components.Energy.Converters.Motor() 
    motor.efficiency             = 0.935
    motor.gearbox_efficiency     = 1.  
    motor.nominal_voltage        = bat.max_voltage *3/4  
    motor.propeller_radius       = rotor.tip_radius    
    motor.no_load_current        = 2.0 
    motor                        = size_optimal_motor(motor,rotor) 
    motor.mass_properties.mass   = nasa_motor(motor.design_torque)
    net.motor                    = motor   

    # Component 5: Payload
    payload                      = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 10. #Watts 
    payload.mass_properties.mass = 1.0 * Units.kg
    net.payload                  = payload

    # Component 6: Avionics
    avionics                     = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw          = 20. #Watts  
    net.avionics                 = avionics

    # Component 7: Miscellaneous Systems 
    sys = SUAVE.Components.Systems.System()
    sys.mass_properties.mass = 5 # kg     
 
    vehicle.append_component(net)   
    
    vehicle.wings['main_wing'].motor_spanwise_locations = np.multiply(
        2./36.25, [2.283 ,-2.283, 2.283, -2.283]) 
    
    return vehicle

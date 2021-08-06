# X57_Mod4.py
#
# Created: Jul 2021, R. Erhard

""" 
Setup file for the X57-Maxwell Mod 4 Electric Aircraft. Configuration
consists of two tip-mounted cruise propellers, along with 12 spanwise 
propellers used for high-lift during takeoff and landing.
"""
 
## ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data 

from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass 
from SUAVE.Methods.Propulsion.electric_motor_sizing  import size_optimal_motor, size_from_mass
from SUAVE.Methods.Propulsion import propeller_design 
 

from copy import deepcopy
import numpy as np 

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'X57_Mod4'    


    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    

    # mass properties
    vehicle.mass_properties.max_takeoff   = 2550. * Units.pounds
    vehicle.mass_properties.takeoff       = 2550. * Units.pounds
    vehicle.mass_properties.max_zero_fuel = 2550. * Units.pounds
    vehicle.mass_properties.cargo         = 0. 

    # envelope properties
    vehicle.envelope.ultimate_load = 5.7
    vehicle.envelope.limit_load    = 3.8

    # basic parameters
    vehicle.reference_area         = 15.45  
    vehicle.passengers             = 4

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.sweeps.leading_edge     = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.areas.reference         = 6.144
    wing.spans.projected         = 31.6 * Units.feet  

    wing.chords.root             = 2.48 * Units.feet  
    wing.chords.tip              = 1.74 * Units.feet  
    wing.chords.mean_aerodynamic = 2.13 * Units.feet   
    wing.taper                   = wing.chords.tip/wing.chords.root

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[2.93, 0., 1.01]]
    wing.aerodynamic_center      = [3.0, 0., 1.01]

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

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.areas.reference         = 3.74 * Units['meters**2']  
    wing.spans.projected         = 3.454  * Units.meter 
    wing.sweeps.quarter_chord    = 12.5 * Units.deg

    wing.chords.root             = 1.397 * Units.meter 
    wing.chords.tip              = 0.762 * Units.meter 
    wing.chords.mean_aerodynamic = 1.09 * Units.meter 
    wing.taper                   = wing.chords.tip/wing.chords.root

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[7.7, 0., 0.25 ]]
    wing.aerodynamic_center      = [7.8, 0., 0.25]

    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False

    wing.dynamic_pressure_ratio  = 0.9

    # add to vehicle
    vehicle.append_component(wing)


    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    

    wing.sweeps.quarter_chord    = 25. * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.areas.reference         = 2.258 * Units['meters**2']  
    wing.spans.projected         = 1.854   * Units.meter 

    wing.chords.root             = 1.6764 * Units.meter 
    wing.chords.tip              = 0.6858 * Units.meter 
    wing.chords.mean_aerodynamic = 1.21 * Units.meter 
    wing.taper                   = wing.chords.tip/wing.chords.root

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[6.75 ,0,  0.623]]
    wing.aerodynamic_center      = [0.508 ,0,0] 

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
    segment                                     = SUAVE.Components.Fuselages.Segment() 
    segment.tag                                 = 'segment_0'  
    segment.percent_x_location                  = 0 
    segment.percent_z_location                  = 0 
    segment.height                              = 0.01 
    segment.width                               = 0.01 
    fuselage.Segments.append(segment)             
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_1'   
    segment.percent_x_location                  = 0.007279116466
    segment.percent_z_location                  = 0.002502014453
    segment.height                              = 0.1669064748
    segment.width                               = 0.2780205877
    fuselage.Segments.append(segment)    

    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_2'   
    segment.percent_x_location                  = 0.01941097724 
    segment.percent_z_location                  = 0.001216095397 
    segment.height                              = 0.3129496403 
    segment.width                               = 0.4365777215 
    fuselage.Segments.append(segment)         

    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_3'   
    segment.percent_x_location                  = 0.06308567604
    segment.percent_z_location                  = 0.007395489231
    segment.height                              = 0.5841726619
    segment.width                               = 0.6735119903
    fuselage.Segments.append(segment)      
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_4'   
    segment.percent_x_location                  = 0.1653761217 
    segment.percent_z_location                  = 0.02891281352 
    segment.height                              = 1.064028777 
    segment.width                               = 1.067200529 
    fuselage.Segments.append(segment)  
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_5'    
    segment.percent_x_location                  = 0.2426372155 
    segment.percent_z_location                  = 0.04214148761 
    segment.height                              = 1.293766653 
    segment.width                               = 1.183058255 
    fuselage.Segments.append(segment)  
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_6'    
    segment.percent_x_location                  = 0.2960174029  
    segment.percent_z_location                  = 0.04705241831  
    segment.height                              = 1.377026712  
    segment.width                               = 1.181540054  
    fuselage.Segments.append(segment)  
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_7'   
    segment.percent_x_location                  = 0.3809404284 
    segment.percent_z_location                  = 0.05313580461 
    segment.height                              = 1.439568345 
    segment.width                               = 1.178218989 
    fuselage.Segments.append(segment)    
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_8'   
    segment.percent_x_location                  = 0.5046854083 
    segment.percent_z_location                  = 0.04655492473 
    segment.height                              = 1.29352518 
    segment.width                               = 1.054390707 
    fuselage.Segments.append(segment)   
    
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_9'   
    segment.percent_x_location                  = 0.6454149933 
    segment.percent_z_location                  = 0.03741966266 
    segment.height                              = 0.8971223022 
    segment.width                               = 0.8501926505   
    fuselage.Segments.append(segment)  
      
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_10'   
    segment.percent_x_location                  = 0.985107095 
    segment.percent_z_location                  = 0.04540283436 
    segment.height                              = 0.2920863309 
    segment.width                               = 0.2012565415  
    fuselage.Segments.append(segment)         
       
    # Segment                                   
    segment                                     = SUAVE.Components.Fuselages.Segment()
    segment.tag                                 = 'segment_11'   
    segment.percent_x_location                  = 1 
    segment.percent_z_location                  = 0.04787575562  
    segment.height                              = 0.1251798561 
    segment.width                               = 0.1206021048 
    fuselage.Segments.append(segment)             
    
    # add to vehicle
    vehicle.append_component(fuselage)   

    #---------------------------------------------------------------------------------------------
    # DEFINE PROPELLER NETWORK AND PROPELLERS
    #---------------------------------------------------------------------------------------------
    # build network    
    net                               = Battery_Propeller() 
    net.number_of_propeller_engines   = 14.                  # 2 during cruise, +12 used during takeoff/landing
    net.nacelle_diameter              = np.array([1.66, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.66]) * Units.feet
    net.engine_length                 = np.array([3., 1., 1., 1., 1., 1., 1.,
                                                  1., 1., 1., 1., 1., 1., 3.]) * Units.feet
    net.identical_propellers          = False
    net.areas                         = Data()
    net.areas.wetted                  = net.nacelle_diameter*net.engine_length + (np.pi*net.nacelle_diameter**2/2)    


    # Component 1 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc
    
    # Component 8 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 500. * Units.kg  
    bat.specific_energy      = 350. * Units.Wh/Units.kg
    bat.resistance           = 0.006
    bat.max_voltage          = 500.
    
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat 
    net.voltage              = bat.max_voltage
        
    
    # -----------------------------------------------------------------------------------
    # Design the tip propellers and corresponding motors
    # -----------------------------------------------------------------------------------
    tip_prop                        = SUAVE.Components.Energy.Converters.Propeller() 
    tip_prop.tag                    = "cruise_propeller"
    tip_prop.number_of_blades       = 3.0
    tip_prop.freestream_velocity    = 174.*Units['mph']    
    tip_prop.angular_velocity       = 2250.  * Units.rpm  
    tip_prop.tip_radius             = 1.523/2
    tip_prop.hub_radius             = 0.1
    tip_prop.design_Cl              = 0.75
    tip_prop.design_altitude        = 8000. * Units.feet
    tip_prop.design_tip_mach        = 0.6
    tip_prop.design_power           = 48100.
    tip_prop.origin                 = [[2.5, 4.97584, 1.01]]        
    tip_prop.rotation               = -1 
    tip_prop.variable_pitch         = True

    tip_prop.airfoil_geometry       =  ['../Vehicles/Airfoils/NACA_4412.txt'] 
    tip_prop.airfoil_polars         = [['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ]]

    tip_prop.airfoil_polar_stations = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]       
    tip_prop                        = propeller_design(tip_prop)
    
    tip_prop_left = deepcopy(tip_prop)
    tip_prop_left.origin   = [[2.5, -4.97584, 1.01]]
    tip_prop_left.rotation = 1 
    
    
    net.propellers.append(tip_prop)
    net.propellers.append(tip_prop_left)
    
    
    #------------------------------------------------------------------
    # Design high lift propellers
    #------------------------------------------------------------------    
    highlift_prop                        = SUAVE.Components.Energy.Converters.Propeller() 
    highlift_prop.tag                    = "high_lift_propeller"
    highlift_prop.number_of_blades       = 5
    highlift_prop.freestream_velocity    = 64.*Units['mph']    
    highlift_prop.angular_velocity       = 4000.  * Units.rpm  
    highlift_prop.tip_radius             = 0.58/2
    highlift_prop.hub_radius             = 0.1
    highlift_prop.design_Cl              = 0.75
    highlift_prop.design_altitude        = 1000. * Units.feet
    highlift_prop.design_tip_mach        = 0.6
    highlift_prop.design_power           = 1400.
    highlift_prop.origin                 = [[2.5, 1.05, 1.01]] 
    highlift_prop.rotation               = -1 
    highlift_prop.variable_pitch         = True

    highlift_prop.airfoil_geometry       =  ['../Vehicles/Airfoils/NACA_4412.txt'] 
    highlift_prop.airfoil_polars         = [['../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_50000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_100000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_200000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_500000.txt' ,
                                    '../Vehicles/Airfoils/Polars/NACA_4412_polar_Re_1000000.txt' ]]    

    highlift_prop.airfoil_polar_stations = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]       
    highlift_prop                        = propeller_design(highlift_prop)    
    
    dist = 0.6
    
    for i in range(12):
        side_factor = -1+2*(i//6) # -1 if on port side, 1 if on starboard side of vehicle
        p           = deepcopy(highlift_prop)
        p.origin    = [[2.5, (1.05+dist*(i%6))*side_factor, 1.01]]  
        p.rotation  = -side_factor
        net.propellers.append(p)
    
    # --------------------------------------------------------------------
    # CORRESPONDING MOTORS
    # --------------------------------------------------------------------
    motor                   = SUAVE.Components.Energy.Converters.Motor()
    etam                    = 0.95
    v                       = bat.max_voltage
    io                      = 2.0 
    start_kv                = 1
    end_kv                  = 25
    # do optimization to find kv or just do a linspace then remove all negative values, take smallest one use 0.05 change
    # essentially you are sizing the motor for a particular rpm which is sized for a design tip mach 
    # this reduces the bookkeeping errors     
    possible_kv_vals        = np.linspace(start_kv,end_kv,(end_kv-start_kv)*20 +1 , endpoint = True) * Units.rpm
    
    
    motor.mass_properties.mass = 10. * Units.kg
    motor.speed_constant       = 0.35 
    motor.no_load_current      = io 
    motor.gear_ratio           = 1. 
    motor.gearbox_efficiency   = 1.  
    
    props = net.propellers
    keys = props.keys()
    for i in range(len(props)):
        p                    = props[list(keys)[i]]
        omeg                 = p.angular_velocity  
        res_kv_vals          = ((v-omeg/possible_kv_vals)*(1.-etam*v*possible_kv_vals/omeg))/io  
        positive_res_vals    = np.extract(res_kv_vals > 0 ,res_kv_vals) 
        res                  = min(positive_res_vals) 
        
        m                    = deepcopy(motor)
        m.resistance         = res
        m.tag                = p.tag+"_motor"
        m.origin             = p.origin
        m.propeller_radius   = p.tip_radius   
        net.propeller_motors.append(m)

    # --------------------------------------------------------------------
    # Component 9 Miscellaneous Systems 
    # --------------------------------------------------------------------
    sys = SUAVE.Components.Systems.System()
    sys.mass_properties.mass = 5 # kg


    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 10. #Watts 
    payload.mass_properties.mass = 1.0 * Units.kg
    net.payload                  = payload

    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 20. #Watts  
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

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    base_config.networks.battery_propeller.pitch_command = 0.
    configs.append(base_config) 


    # done!
    return configs


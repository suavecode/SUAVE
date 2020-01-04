# X57_Maxwell.py
#
# Created: Dec 2019, M. Clarke

""" setup file for the X 57 Maxwell all-electric general aviation vehicle
"""
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

import numpy as np 
from SUAVE.Core import Data

from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_kv

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------
def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle     = SUAVE.Vehicle()
    vehicle.tag = 'X57_Maxwell' 

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

    wing.sweeps.leading_edge     = 0 * Units.degrees 
    wing.thickness_to_chord      = 0.17
    wing.span_efficiency         = 0.95
    wing.areas.reference         = 15.7222 * Units['meters**2']  
    wing.spans.projected         = 11.072 * Units.meter   
    wing.chords.root             = 1.4391 * Units.meter  
    wing.chords.tip              = 0.84 * Units.meter   
    wing.taper                   = wing.chords.root/wing.chords.tip
    wing.chords.mean_aerodynamic = wing.chords.root * 2/3 * (( 1 + wing.taper  + wing.taper **2 ) /( 1 + wing.taper  )) 
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root             = 4 * Units.degrees
    wing.twists.tip              = 0 * Units.degrees 
    wing.origin                  = [2.9436, 0., 0.75 ] 
    wing.aerodynamic_center      = [3.125, 0., 0.75]  
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = True

    wing.dynamic_pressure_ratio  = 1.0
    
    # Root
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'root'
    segment.percent_span_location = 0.0
    segment.twist                 = 4. * Units.deg
    segment.root_chord_percent    = 1.
    segment.thickness_to_chord    = 0.17
    segment.dihedral_outboard     = 0 * Units.degrees
    segment.sweeps.leading_edge   = 0 * Units.degrees 
    wing.Segments.append(segment)    
    
    
    # Yehudi
    segment = SUAVE.Components.Wings.Segment()
    segment.tag                   = 'mid_segment'
    segment.percent_span_location = 0.5690 
    segment.twist                 = (wing.twists.root*(1-segment.percent_span_location)) * Units.deg
    segment.root_chord_percent    = 1.0 
    segment.thickness_to_chord    = 0.12
    segment.dihedral_outboard     = 0 * Units.degrees
    segment.sweeps.leading_edge   = 0 * Units.degrees 
    wing.Segments.append(segment)
    
    # Add a tip
    segment = SUAVE.Components.Wings.Segment() 
    segment.tag                   = 'Tip'
    segment.percent_span_location = 1.
    segment.twist                 = 0. * Units.deg
    segment.root_chord_percent    = 0.5836 
    segment.thickness_to_chord    = 0.1
    segment.dihedral_outboard     = 0.
    segment.sweeps.quarter_chord  = 0. 
    wing.Segments.append(segment)     
   
    # add to vehicle
    vehicle.append_component(wing) 


    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.15
    wing.span_efficiency         = 0.95
    wing.areas.reference         = 2.5542 * Units['meters**2']  
    wing.spans.projected         = 3.3 * Units.meter 
    wing.sweeps.quarter_chord    = 0 * Units.deg 
    wing.chords.root             = 0.774 * Units.meter 
    wing.chords.tip              = 0.774 * Units.meter  
    wing.taper                   = wing.chords.root/wing.chords.tip
    wing.chords.mean_aerodynamic = wing.chords.root * 2/3 * (( 1 + wing.taper  + wing.taper **2 ) /( 1 + wing.taper  )) 
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees 
    wing.origin                  = [7.686, 0., 0.09] 
    wing.aerodynamic_center      = [7.9 , 0., 0.09] 
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

    wing.sweeps.quarter_chord    = 35 * Units.deg
    wing.thickness_to_chord      = 0.15
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 1.752 * Units['meters**2']  
    wing.spans.projected         = 1.6 * Units.meter  
    wing.chords.root             = 1.58 * Units.meter 
    wing.chords.tip              = 0.61 * Units.meter     
    wing.taper                   = wing.chords.root/wing.chords.tip
    wing.chords.mean_aerodynamic = wing.chords.root * 2/3 * (( 1 + wing.taper  + wing.taper **2 ) /( 1 + wing.taper  )) 
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference 
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees 
    wing.origin                  = [6.74,0,0.0]
    wing.aerodynamic_center      = [0,0,0] 
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
    fuselage.tag = 'fuselage'

    fuselage.seats_abreast         = 2.

    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.

    fuselage.lengths.nose          = 2.35
    fuselage.lengths.tail          = 8.7 - 4.52
    fuselage.lengths.cabin         = 4.52 - 2.35
    fuselage.lengths.total         = 8.7  
    fuselage.width                 = 1.2 
    fuselage.heights.maximum       = 1.42
    fuselage.heights.at_quarter_length          = 1.24 
    fuselage.heights.at_three_quarters_length   = 1.08 
    fuselage.heights.at_wing_root_quarter_chord = 0.52

    fuselage.areas.side_projected  = 7.33 * Units['meters**2'] 
    fuselage.areas.wetted          = 125.25 * Units['meters**2'] 
    fuselage.areas.front_projected = 1.508 * Units['meters**2']   
    fuselage.effective_diameter    = 1.2   
    
    # add to vehicle
    vehicle.append_component(fuselage)

    #---------------------------------------------------------------------------------------------
    # DEFINE PROPELLER
    #---------------------------------------------------------------------------------------------
    # build network    
    net = Battery_Propeller()
    net.number_of_engines       = 2.
    net.nacelle_diameter        = 42 * Units.inches
    net.engine_length           = 0.01 * Units.inches
    net.areas                   = Data() 
    net.areas.wetted            = 0.01*(2*np.pi*0.01/2)    
    net.voltage                 = 500.
    
    # Component 1 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc

    # Component 2 the Propeller 
    prop = SUAVE.Components.Energy.Converters.Propeller() 
    prop.number_blades       = 2.0
    prop.freestream_velocity = 135.*Units['mph']    
    prop.angular_velocity    = 1300.  * Units.rpm # 2400
    prop.tip_radius          = 0.9652
    prop.hub_radius          = 0.1235  
    prop.design_Cl           = 0.8
    prop.design_altitude     = 12000. * Units.feet 
    prop.design_thrust       = 1000. 
    prop.design_power        = 0. * Units.watts 
    prop.origin              = [[2.94,1.59,0.75],[2.94,-1.59,0.75]]  # propeller origin          
    prop.rotation            = [1,-1]                                # rotation of propeller blade [1 = clockwise, -1 = anti-clockwise]
    prop                     = propeller_design(prop)    
    net.propeller            = prop    

    # Component 3 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 500. * Units.kg  
    bat.specific_energy      = 350. * Units.Wh/Units.kg
    bat.resistance           = 0.006
    bat.max_voltage          = 400.    
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat
     
    # Component 4 the Motor
    motor                    = SUAVE.Components.Energy.Converters.Motor()
    etam                     = 0.95
    v                        = bat.max_voltage  
    omeg                     = prop.angular_velocity  
    io                       = 4.0 
    start_kv                 = 1
    end_kv                   = 50
    # do optimization to find kv or just do a linspace then remove all negative values, take smallest one use 0.05 change
    # essentially you are sizing the motor for a particular rpm which is sized for a design tip mach 
    # this reduces the bookkeeping errors     
    possible_kv_vals            = np.linspace(start_kv,end_kv,(end_kv-start_kv)*20 +1 , endpoint = True) * Units.rpm
    res_kv_vals                 = ((v-omeg/possible_kv_vals)*(1.-etam*v*possible_kv_vals/omeg))/io  
    positive_res_vals           = np.extract(res_kv_vals > 0 ,res_kv_vals) 
    kv_idx                      = np.where(res_kv_vals == min(positive_res_vals))[0][0]   
    kv                          = possible_kv_vals[kv_idx]  
    res                         = max(positive_res_vals)  
    motor.mass_properties.mass  = 10. * Units.kg
    motor.origin                = prop.origin  
    motor.propeller_radius      = prop.tip_radius   
    motor.speed_constant        = 0.35 
    motor.resistance            = res
    motor.no_load_current       = io 
    motor.gear_ratio            = 1. 
    motor.gearbox_efficiency    = 1. # Gear box efficiency     
    net.motor                   = motor 

    # Component 5 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 10. #Watts 
    payload.mass_properties.mass = 1.0 * Units.kg
    net.payload                  = payload

    # Component 6 the Avionics
    avionics            = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 20. #Watts  
    net.avionics        = avionics      
 
    vehicle.append_component(net)          

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------ 
    return vehicle

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

def configs_setup(vehicle):

    # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------

    configs = SUAVE.Components.Configs.Config.Container()

    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
 
     # done!
    return configs

# ----------------------------------------------------------------------
#   Sizing for the Vehicle Configs
# ----------------------------------------------------------------------
def simple_sizing(configs):

    base = configs.base
    base.pull_base()

    # wing areas
    for wing in base.wings:
        wing.areas.wetted   = 1.75 * wing.areas.reference
        wing.areas.exposed  = 0.8  * wing.areas.wetted
        wing.areas.affected = 0.6  * wing.areas.wetted


    # diff the new data
    base.store_diff()


    # done!
    return


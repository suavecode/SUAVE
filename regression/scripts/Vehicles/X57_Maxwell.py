# Boeing_737.py
#
# Created: Feb 2020, M. Clarke

""" setup file for the X57-Maxwell Electric Aircraft 
"""
 
## ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np   
from SUAVE.Core import Data 
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design 
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_optimal_motor
from SUAVE.Methods.Weights.Correlations.Propulsion import nasa_motor

# ----------------------------------------------------------------------
#   Define the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():

    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    

    vehicle = SUAVE.Vehicle()
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

    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.areas.reference         = 15.45 * Units['meters**2']  
    wing.spans.projected         = 11. * Units.meter  

    wing.chords.root             = 1.67 * Units.meter  
    wing.chords.tip              = 1.14 * Units.meter  
    wing.chords.mean_aerodynamic = 1.47 * Units.meter   
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 1.5 * Units.degrees

    wing.origin                  = [[2.032, 0., 0.784]]
    wing.aerodynamic_center      = [0.558, 0., 0.784]

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
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[6.248, 0., 0 ]]
    wing.aerodynamic_center      = [0.508, 0., 0.]

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
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [[6.01 ,0,  0.623]]
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
    fuselage.tag = 'fuselage'

    fuselage.seats_abreast         = 2.

    fuselage.fineness.nose         = 1.6
    fuselage.fineness.tail         = 2.

    fuselage.lengths.nose          = 60.  * Units.inches
    fuselage.lengths.tail          = 161. * Units.inches
    fuselage.lengths.cabin         = 105. * Units.inches
    fuselage.lengths.total         = 326. * Units.inches
    fuselage.lengths.fore_space    = 0.
    fuselage.lengths.aft_space     = 0.    

    fuselage.width                 = 42. * Units.inches

    fuselage.heights.maximum       = 62. * Units.inches
    fuselage.heights.at_quarter_length          = 62. * Units.inches
    fuselage.heights.at_three_quarters_length   = 62. * Units.inches
    fuselage.heights.at_wing_root_quarter_chord = 23. * Units.inches

    fuselage.areas.side_projected  = 8000.  * Units.inches**2.
    fuselage.areas.wetted          = 30000. * Units.inches**2.
    fuselage.areas.front_projected = 42.* 62. * Units.inches**2.

    fuselage.effective_diameter    = 50. * Units.inches


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


    # Component 1 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc

    # Component 2 the Propeller 
    #------------------------------------------------------------------
    # Design Rotors and Propellers
    #------------------------------------------------------------------
    # atmosphere and flight conditions for propeller/rotor design
    speed_of_sound               = 340
    g              = 9.81                                   # gravitational acceleration 
    S              = vehicle.reference_area                 # reference area 
    rho            = 1.2                                    # reference density
    fligth_CL      = 0.75                                   # cruise target lift coefficient 
    AR             = vehicle.wings.main_wing.aspect_ratio   # aspect ratio 
    Cd0            = 0.06                                   # profile drag
    Cdi            = fligth_CL**2/(np.pi*AR*0.98)           # induced drag
    Cd             = Cd0 + Cdi                              # total drag
    V_inf          = 140.* Units['mph']                     # freestream velocity 
    Drag           = S * (0.5*rho*V_inf**2 )*Cd             # cruise drag 
    
    prop = SUAVE.Components.Energy.Converters.Propeller() 

    prop.number_blades       = 2.0
    prop.freestream_velocity = V_inf
    prop.design_tip_mach     = 0.5 
    prop.tip_radius          = 76./2. * Units.inches
    prop.hub_radius          = 8.     * Units.inches    
    prop.angular_velocity    = prop.design_tip_mach*speed_of_sound/prop.tip_radius  
    prop.design_Cl           = 0.7
    prop.design_altitude     = 12000. * Units.feet
    prop.design_thrust       = Drag/net.number_of_engines
    prop.origin              = [[2.,2.5,0.784],[2.,-2.5,0.784]]                 
    prop.symmetry            = True
    prop                     = propeller_design(prop)    
    net.propeller            = prop    
      
    velocity    = prop.freestream_velocity
    density     = 1.21  
    rps         = prop.angular_velocity/(Units.rpm*60) # rev per sec 
    design_Cq   = prop.design_power /(2*np.pi*density*(rps**3)*((prop.tip_radius*2)**5))
    des_Cp      = 2*np.pi* design_Cq
    des_etap    = (prop.design_thrust*velocity)/(2*np.pi*rps*prop.design_torque )  
    print('\nPropeller Design Attributes')
    print('Design Thrust         = ' + str(round(prop.design_thrust ,4)))
    print('Design Torque         = ' + str(round(prop.design_torque,4)))
    print('Design Power          = ' + str(round(prop.design_power ,4)))
    print('Power Coefficient     = ' + str(round(des_Cp,4)))
    print('Design Efficiency     = ' + str(round(des_etap,4))) 
    
    # Component 8 the Battery
    bat                      = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 500. * Units.kg  
    bat.specific_energy      = 350. * Units.Wh/Units.kg
    bat.resistance           = 0.006
    bat.max_voltage          = 500.
    
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat 
    net.voltage              = bat.max_voltage
    
    # Component 9 Miscellaneous Systems 
    sys = SUAVE.Components.Systems.System()
    sys.mass_properties.mass = 5  
    
    #------------------------------------------------------------------
    # Design Motors
    #------------------------------------------------------------------
    # Propeller  motor
    # Component 4 the Motor
    motor                         = SUAVE.Components.Energy.Converters.Motor() 
    motor.mass_properties.mass    = nasa_motor(prop.design_torque) 
    motor.efficiency              = 0.95  
    motor.gear_ratio              = 1.0
    motor.gearbox_efficiency      = 1.0  
    motor.no_load_current         = 1.0     
    motor.resistance              = 0.1
    motor.nominal_voltage         = bat.max_voltage*0.9
    motor                         = size_optimal_motor(motor,prop)   
    net.motor                     = motor  
    print('Motor Speed Constant')
    print("\nMotor Speed Constant = " + str(round(motor.speed_constant, 4)))
                                  
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
    configs.append(base_config) 


    # done!
    return configs


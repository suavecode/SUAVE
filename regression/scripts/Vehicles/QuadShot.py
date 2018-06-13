# vtol_UAV.py
# 
# Created:  Jan 2016, E. Botero
# Modified: 

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data

import numpy as np
import pylab as plt
import matplotlib
import copy, time

from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_kv
import cProfile, pstats, StringIO
#from SUAVE.Components.Energy.Processes.propeller_map import propeller_map

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------

def vehicle_setup():
    
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------    
    
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'Quadshot'
    
    # ------------------------------------------------------------------
    #   Vehicle-level Properties
    # ------------------------------------------------------------------    
    # mass properties
    vehicle.mass_properties.takeoff         = 0.82 * Units.kg
    vehicle.mass_properties.operating_empty = 0.82 * Units.kg
    vehicle.mass_properties.max_takeoff     = 0.82 * Units.kg
    
    # basic parameters
    vehicle.reference_area                  = 0.1668 
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------   

    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    
    wing.areas.reference         = vehicle.reference_area
    wing.spans.projected         = 1.03 * Units.m
    wing.aspect_ratio            = (wing.spans.projected**2)/wing.areas.reference 
    wing.sweep                   = 5.0 * Units.deg
    wing.symmetric               = True
    wing.thickness_to_chord      = 0.12
    wing.taper                   = 1.0
    wing.vertical                = False
    wing.high_lift               = False
    wing.dynamic_pressure_ratio  = 1.0
    wing.chords.mean_aerodynamic = 0.162 * Units.m
    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees
    wing.highlift                = False  
    wing.vertical                = False
    
    # add to vehicle
    vehicle.append_component(wing)    

    #------------------------------------------------------------------
    # Propulsor
    #------------------------------------------------------------------
    
    # build network
    net = Battery_Propeller()
    net.number_of_engines = 4.
    net.nacelle_diameter  = 27. * Units.mm
    net.engine_length     = 14. * Units.mm
    net.voltage           = 12.3
    net.thrust_angle      = 0. * Units.degrees
    net.use_surrogate     = False
    
    # Component 1 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc
    
    # Component 2 the Propeller
    
    # Design the Propeller
    prop_attributes = Data()
    prop_attributes.number_blades       = 2.0
    prop_attributes.freestream_velocity = 15.0 # freestream m/s
    prop_attributes.angular_velocity    = 7500. * Units['rpm']
    prop_attributes.tip_radius          = 4.   * Units.inch
    prop_attributes.hub_radius          = 0.125 * Units.inch
    prop_attributes.design_Cl           = 0.7
    prop_attributes.design_altitude     = 0.1 * Units.km
    prop_attributes.design_thrust       = 0.0#.70*9.81
    prop_attributes.design_power        = 200. * Units.watts
    prop_attributes                     = propeller_design(prop_attributes)
    
    #points = Data()
    #points.altitudes  = [0.,1. *Units.km]
    #points.velocities = [0.1,3.,15.,20.]
    #points.omega      = [500.*Units['rpm'],1000.*Units['rpm'],2000.*Units['rpm'],3000.*Units['rpm'],4000.*Units['rpm'],5000.*Units['rpm'],6000.*Units['rpm'],7000.*Units['rpm']]
    
    #prop = propeller_map(prop,points)    

    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes = prop_attributes
    net.propeller        = prop

    # Component 3 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor()
    kv    = 1500. * Units['rpm'] # RPM/volt converted to (rad/s)/volt 
    motor = size_from_kv(motor, kv)
    motor.gear_ratio           = 1.  # Gear ratio
    motor.gearbox_efficiency   = 1.  # Gear box efficiency
    motor.expected_current     = 10. # Expected current
    motor.propeller_radius     = prop_attributes.tip_radius
    motor.propeller_Cp         = prop_attributes.Cp  
    net.motor                  = motor        
    
    # Component 4 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 0. #Watts 
    payload.mass_properties.mass = 0.0 * Units.kg
    net.payload                  = payload
    
    # Component 5 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 2. #Watts  
    net.avionics        = avionics      

    # Component 6 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 0.17 * Units.kg
    bat.specific_energy      = 175.*Units.Wh/Units.kg
    bat.resistance           = 0.00#3
    bat.max_voltage          = 11.1
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery              = bat
    
    # add the solar network to the vehicle
    vehicle.append_component(net)  

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
    
    return configs
# eVTOL_Weights_Buildup_Regression.py
#
# Created: Feb, 2018, J. Smart

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import sys
import SUAVE
import numpy as np
import copy as cp
import pprint as pp
from SUAVE.Methods.Weights.Buildups.Electric_Helicopter import empty as electricHelicopterEmpty
from SUAVE.Methods.Weights.Buildups.Electric_Stopped_Rotor import empty as electricStoppedRotorEmpty
from SUAVE.Methods.Weights.Buildups.Electric_Tiltrotor import empty as electricTiltrotorEmpty
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Core import (
    Data, Container, Units,
)


#-------------------------------------------------------------------------------
# Define Base Vehicle
#-------------------------------------------------------------------------------

def vehicle_setup():
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'eVTOL Weight Test'
    
    #---------------------------------------------------------------------------
    # Vehicle-Level Properties
    #---------------------------------------------------------------------------
    
    vehicle.mass_properties.takeoff         = 840.  * Units.kg
    vehicle.mass_properties.operating_empty = 640.  * Units.kg
    vehicle.mass_properties.max_takeoff     = 1000. * Units.kg
    
    vehicle.reference_area                  = 12.   * Units['meter**2']
    
    #---------------------------------------------------------------------------
    # Fuselage Properties Properties
    #---------------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    fuselage.lengths.total      = 5.5 * Units.meter
    fuselage.width              = 2   * Units.meter
    fuselage.heights.maximum    = 1.5 * Units.meter
    
    vehicle.append_component(fuselage)
    
    #---------------------------------------------------------------------------
    # Energy Network Properties
    #---------------------------------------------------------------------------
    
    net = Battery_Propeller()
    net.voltage = 375
    
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency  = 0.95
    net.esc         = esc
    
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw              = 0.
    payload.mass_properties.mass    = 200 * Units.kg
    net.payload                     = payload

    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 20.
    net.avionics        = avionics
    
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass    = 300. * Units.kg
    bat.specific_energy         = 300. * Units.Wh/Units.kg
    bat.resistance              = 0.006
    bat.max_voltage             = 400
    initialize_from_mass(bat,bat.mass_properties.mass)
    net.battery                 = bat
    
    vehicle.append_component(net)
    
    #---------------------------------------------------------------------------
    # Main Wing
    #---------------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main wing'
    
    wing.aspect_ratio               = 4.
    wing.thickness_to_chord         = 0.15
    wing.taper                      = 1.
    
    wing.spans.projected            = 6. * Units.meter
    
    wing.areas.reference            = 12.* Units['meter**2']
    
    wing.chords.root                = 2. * Units.meter
    wing.chords.tip                 = 2. * Units.meter
    wing.chords.mean_aerodynamic    = 2. * Units.meter
    
    wing.sweeps.leading_edge        = 0. * Units.degrees
    wing.sweeps.half_chord          = 0. * Units.degrees
    
    wing.motor_spanwise_locations   = [0.5, 0.5]
    wing.winglet_fraction           = 0.
    
    wing.high_lift                  = True
    
    vehicle.append_component(wing)
    
    #---------------------------------------------------------------------------
    # Secondary Wing
    #---------------------------------------------------------------------------    
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'secondary wing'
    
    wing.aspect_ratio               = 4.
    wing.thickness_to_chord         = 0.15
    wing.taper                      = 1.
    
    wing.spans.projected            = 6. * Units.meter
    
    wing.areas.reference            = 12.* Units['meter**2']
    
    wing.chords.root                = 2. * Units.meter
    wing.chords.tip                 = 2. * Units.meter
    wing.chords.mean_aerodynamic    = 2. * Units.meter
    
    wing.sweeps.leading_edge        = 0. * Units.degrees
    wing.sweeps.half_chord          = 0. * Units.degrees
    
    wing.high_lift                  = True
    
    wing.motor_spanwise_locations   = [0.5, 0.5]
    wing.winglet_fraction           = 0.
    
    vehicle.append_component(wing)
    
    return vehicle

def configs_setup(vehicle):
    
    #---------------------------------------------------------------------------
    # Initialize Configurations
    #---------------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Config.Container()
    
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    #---------------------------------------------------------------------------
    # Electric Helicopter Configuration
    #---------------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'electric_helicopter'
    
    config.propulsors.network.number_of_engines = 1
    
    prop_attributes = Data()
    prop_attributes.number_blades       = 4.0
    prop_attributes.freestream_velocity = 150.  * Units['meter/second']
    prop_attributes.angular_velocity    = 3500. * Units['rpm']
    prop_attributes.tip_radius          = 3800. * Units.mm
    prop_attributes.hub_radius          = 500.  * Units.mm
    prop_attributes.design_Cl           = 0.7
    prop_attributes.design_altitude     = 1.    * Units.km
    prop_attributes.design_thrust       = 1600. * 9.81 * Units.newtons
    prop_attributes.design_power        = 0.    * Units.watts
    prop_attributes                     = propeller_design(prop_attributes)
       
    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes    = prop_attributes
    prop.origin             = [0.,0.,0.]
    config.propulsors.network.propeller    = prop
    
    configs.append(config)
    
    #---------------------------------------------------------------------------
    # Electric Stopped Rotor Configuration
    #---------------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'electric_stopped_rotor'
    
    config.propulsors.network.number_of_engines = 8
    
    prop_attributes = Data()
    prop_attributes.number_blades       = 4.0
    prop_attributes.freestream_velocity = 150.  * Units['meter/second']
    prop_attributes.angular_velocity    = 3500. * Units['rpm']
    prop_attributes.tip_radius          = 800.  * Units.mm #608
    prop_attributes.hub_radius          = 150. * Units.mm
    prop_attributes.design_Cl           = 0.7
    prop_attributes.design_altitude     = 1. * Units.km
    prop_attributes.design_thrust       = 200. * 9.81 * Units.newtons
    prop_attributes.design_power        = 0. * Units.watts
    prop_attributes                     = propeller_design(prop_attributes)
       
    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes    = prop_attributes
    prop.origin             = [0.,0.,0.]
    config.propulsors.network.propeller    = prop
    
    thrust_prop_attributes = cp.deepcopy(prop_attributes)
    thrust_prop_attributes.number_blades = 2.0
    thrust_prop = SUAVE.Components.Energy.Converters.Propeller()
    thrust_prop.prop_attributes = thrust_prop_attributes
    config.propulsors.network.thrust_propeller = thrust_prop
    
    configs.append(config)

    #---------------------------------------------------------------------------
    # Electric Tiltrotor Configuration
    #---------------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'electric_tiltrotor'
    
    config.propulsors.network.number_of_engines = 8
    
    prop_attributes = Data()
    prop_attributes.number_blades       = 4.0
    prop_attributes.freestream_velocity = 150.  * Units['meter/second']
    prop_attributes.angular_velocity    = 3500. * Units['rpm']
    prop_attributes.tip_radius          = 800.  * Units.mm #608
    prop_attributes.hub_radius          = 150.  * Units.mm
    prop_attributes.design_Cl           = 0.7
    prop_attributes.design_altitude     = 1.    * Units.km
    prop_attributes.design_thrust       = 200.  * 9.81 * Units.newtons
    prop_attributes.design_power        = 0.    * Units.watts
    prop_attributes                     = propeller_design(prop_attributes)    
       
    prop = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes    = prop_attributes
    prop.origin             = [0.,0.,0.]
    config.propulsors.network.propeller    = prop
    
    configs.append(config)
    
    return configs

def full_setup():
    
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)
    
    #---------------------------------------------------------------------------
    # Reference Order: Helicopter, Stopped Rotor, Tiltrotor
    #---------------------------------------------------------------------------
    
    referenceWeights = [
    {'avionics': 15.0,
    'battery': 300.0,
    'brs': 16.0,
    'empty': 687.81033902919182,
    'fuselage': 72.297963706434928,
    'hub': 40.0,
    'landing_gear': 20.0,
    'motors': 20.0,
    'payload': 200,
    'rotor': 59.574357372095562,
    'seats': 30.0,
    'servos': 5.2,
    'structural': 191.87232107853049,
    'tail_rotor': 10.232485041652151,
    'total': 874.81033902919182,
    'transmission': 38.75307946983569,
    'wiring': 47.209805311643777},
    {'avionics': 15.0,
    'battery': 300.0,
    'brs': 16.0,
    'empty': 729.4715536480071,
    'fuselage': 72.297963706434928,
    'hubs': 16.0,
    'landing_gear': 20.0,
    'lift_rotors': 8.9154453601562373,
    'main_wing': 42.734077092628688,
    'motors': 80.0,
    'payload': 200,
    'seats': 30.0,
    'sec_wing': 42.734077092628688,
    'servos': 5.2,
    'structural': 203.73778195497309,
    'thrust_rotors': 1.0562187031245327,
    'total': 929.4715536480071,
    'wiring': 13.218175906851478},
    {'avionics': 15.0,
    'battery': 300.0,
    'brs': 16.0,
    'empty': 737.1097130745701,
    'fuselage': 72.297963706434928,
    'hubs': 16,
    'landing_gear': 20.0,
    'lift_rotors': 8.9154453601562373,
    'main_wing': 42.734077092628688,
    'motors': 80,
    'payload': 200,
    'rotor_servos': 8,
    'seats': 30.0,
    'sec_wing': 42.734077092628688,
    'servos': 5.2,
    'structural': 202.68156325184856,
    'total': 937.1097130745701,
    'wiring': 13.218175906851478}
    ]
    
    
    return vehicle, configs, referenceWeights

def check_results(referenceWeights, refactoredWeights):
    
    errors = 0
    i = 0
    
    for weightDict in referenceWeights:
        for k, v in weightDict.items():
            try:
                refVal = referenceWeights[i].get(k)
                newVal = refactoredWeights[i].get(k)
                err = (newVal-refVal)/refVal
                if (np.abs(err) < 1e-6):
                    pass
                else:
                    print('Reference Check Failed: Dictionary {}, Value {}.\n'.format(i+1, k))
                    print('The reference value is {}, the new value is {}.\n'.format(refVal, newVal))
                    errors += 1
            except KeyError:
                print("The {} value does not appear in the refactored weights".format(k))
        
        i += 1
        
    if errors == 0:
        print('Regression Test Passed.')
    else:
        print('Regression Test Failed with {} errors'.format(errors))
        raise  ValueError


def main():
    
    vehicle, configs, referenceWeights = full_setup()
    
    print("Running Regression Test of eVTOL Weight Buildup Methods.\n")
    
    electricHelicopterEmptyWeights = electricHelicopterEmpty(configs["electric_helicopter"])
    electricStoppedRotorEmptyWeights = electricStoppedRotorEmpty(configs["electric_stopped_rotor"])
    electricTiltrotorEmptyWeights = electricTiltrotorEmpty(configs["electric_tiltrotor"])

    refactoredWeights = [electricHelicopterEmptyWeights,
                         electricStoppedRotorEmptyWeights,
                         electricTiltrotorEmptyWeights]

    check_results(referenceWeights, refactoredWeights)

if __name__ == '__main__':
    main()
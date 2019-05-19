# Cessna_172.py
#
# Created:  Feb 2017, M. Vegh 
# Modified: Feb 2018, M. Vegh 
# Modified: May 2019, M. Clarke

""" setup file fora modified Cessna_172 (2 propeller), current values only used to test General Aviation script
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units , Data 
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
 

def vehicle_setup():
    # ------------------------------------------------------------------
    #   Initialize the Vehicle
    # ------------------------------------------------------------------
    vehicle                               = SUAVE.Vehicle()
    vehicle.tag                           = 'Cessna_172_SP'    
    
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
    vehicle.reference_area         = 174. * Units.feet**2       
    vehicle.passengers             = 4

    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    
    wing.sweeps.quarter_chord    = 0.0 * Units.deg
    wing.thickness_to_chord      = 0.12
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 174. * Units.feet**2
    wing.spans.projected         = 36.  * Units.feet + 1. * Units.inches

    wing.chords.root             = 66. * Units.inches
    wing.chords.tip              = 45. * Units.inches
    wing.chords.mean_aerodynamic = 58. * Units.inches # Guess
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 3.0 * Units.degrees
    wing.twists.tip              = 1.5 * Units.degrees

    wing.origin                  = [80.* Units.inches,0,0] 
    wing.aerodynamic_center      = [22.* Units.inches,0,0]

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
    wing.span_efficiency         = 0.95
    wing.areas.reference         = 5800. * Units.inches**2
    wing.spans.projected         = 136.  * Units.inches

    wing.chords.root             = 55. * Units.inches
    wing.chords.tip              = 30. * Units.inches
    wing.chords.mean_aerodynamic = 43. * Units.inches # Guess
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [246.* Units.inches,0,0] 
    wing.aerodynamic_center      = [20.* Units.inches,0,0]
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
    wing.span_efficiency         = 0.9
    wing.areas.reference         = 3500. * Units.inches**2
    wing.spans.projected         = 73.   * Units.inches

    wing.chords.root             = 66. * Units.inches
    wing.chords.tip              = 27. * Units.inches
    wing.chords.mean_aerodynamic = 48. * Units.inches # Guess
    wing.taper                   = wing.chords.root/wing.chords.tip

    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.origin                  = [237.* Units.inches,0,  0.623] 
    wing.aerodynamic_center      = [20.* Units.inches,0,0] 

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
    
    # ------------------------------------------------------------------
    #   Piston Propeller Network
    # ------------------------------------------------------------------    
    
    # build network
    net = SUAVE.Components.Energy.Networks.Internal_Combustion_Propeller()
    net.number_of_engines = 2.
    net.nacelle_diameter  = 42 * Units.inches
    net.engine_length     = 0.01 * Units.inches
    net.areas             = Data()
    net.rated_speed       = 2700. * Units.rpm
    net.areas.wetted      = 0.01

    # Component 1 the engine
    net.engine = SUAVE.Components.Energy.Converters.Internal_Combustion_Engine()
    net.engine.sea_level_power    = 180. * Units.horsepower
    net.engine.flat_rate_altitude = 0.0
    net.engine.speed              = 2700. * Units.rpm
    net.engine.BSFC               = 0.52


    # Design the Propeller    
    prop  = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades       = 2.0
    prop.freestream_velocity = 135.*Units['mph']    
    prop.angular_velocity    = 1250.  * Units.rpm
    prop.tip_radius          = 76./2. * Units.inches
    prop.hub_radius          = 8.     * Units.inches
    prop.design_Cl           = 0.8
    prop.design_altitude     = 12000. * Units.feet
    prop.design_thrust       = 0.0
    prop.design_power        = .32 * 180. * Units.horsepower
    prop                     = propeller_design(prop)
    prop.origin = [[2.,2.5,0.]] 
    net.propeller        = prop

    # add the network to the vehicle
    vehicle.append_component(net)      
    
    # ------------------------------------------------------------------
    #  Landing Gear
    # ------------------------------------------------------------------
    landing_gear           = SUAVE.Components.Landing_Gear.Landing_Gear()
    main_gear              = SUAVE.Components.Landing_Gear.Main_Landing_Gear()
    nose_gear              = SUAVE.Components.Landing_Gear.Nose_Landing_Gear()
    main_gear.strut_length = 12. * Units.inches #guess based on picture
    nose_gear.strut_length = 6. * Units.inches 
    landing_gear.main      = main_gear
    landing_gear.nose      = nose_gear
    
    #add to vehicle
    vehicle.landing_gear   = landing_gear    

    # ------------------------------------------------------------------
    #  Fuel
    # ------------------------------------------------------------------          
    #define fuel weight needed to size fuel system
    fuel                                   = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    fuel.mass_properties                   = SUAVE.Components.Mass_Properties()
    fuel.mass_properties.mass              = vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel
    fuel.number_of_tanks                   = 1.
    fuel.origin                            = vehicle.wings['main_wing'].origin   
    fuel.internal_volume                   = fuel.mass_properties.mass/fuel.density #all of the fuel volume is internal
    fuel.mass_properties.center_of_gravity = vehicle.wings['main_wing'].mass_properties.center_of_gravity
    vehicle.fuel                           = fuel
 
    # ------------------------------------------------------------------
    #  Avionics
    # ------------------------------------------------------------------    
    Wuav                                   = 2. * Units.lbs
    avionics                               = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.mass_properties.uninstalled   = Wuav
    vehicle.avionics                       = avionics

    # ------------------------------------------------------------------
    #   Vehicle Definition Complete
    # ------------------------------------------------------------------



  
    return vehicle
  
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
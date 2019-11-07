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

    vehicle     = SUAVE.Vehicle()
    vehicle.tag = 'Cessna_172_SP'
    GTOW                                  = 2550. * Units.pounds     
    vehicle.mass_properties.max_takeoff   = GTOW 
    vehicle.mass_properties.empty         = 1669. * Units.pounds
    vehicle.mass_properties.max_zero_fuel = GTOW 
    vehicle.envelope.ultimate_load        = 3.8    
    vehicle.envelope.limit_load           = 1.      

    #define fuel weight needed to size fuel system
    fuel                            = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    fuel.mass_properties            = SUAVE.Components.Mass_Properties()
    fuel.mass_properties.mass       = 319 *Units.lbs
    fuel.number_of_tanks            = 1.
    fuel.internal_volume            = fuel.mass_properties.mass/fuel.density #all of the fuel volume is internal
    vehicle.fuel                    = fuel

    propulsors                      = SUAVE.Components.Propulsors.Propulsor() #use weights for the IC engine  
    propulsors.tag                  = 'internal_combustion'
    propulsors.rated_power          = 110 *Units.kW # engine correlation is really off; compared weight breakdown to Roskam, who bookkept weights differently
    propulsors.number_of_engines    = 1.
    vehicle.append_component(propulsors)
    
    cruise_speed                    = 140. *Units['mph']
    altitude                        = 13500. * Units.ft
    atmo                            = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    freestream                      = atmo.compute_values (0.)
    freestream0                     = atmo.compute_values (altitude)
    mach_number                     = (cruise_speed/freestream.speed_of_sound)[0][0]

    vehicle.passengers              = 4.  #including pilot                           # Number of passengers
    vehicle.mass_properties.cargo   = 0.  * Units.kilogram            # Mass of cargo
    vehicle.reference_area          = 174. * Units.feet**2      # Wing gross area in square meters
    vehicle.design_dynamic_pressure = ( .5 *freestream0.density*(cruise_speed*cruise_speed))[0][0]
    vehicle.design_mach_number      =  mach_number

    #main wing
    wing                           = SUAVE.Components.Wings.Wing()
    wing.tag                       = 'main_wing'
    wing.areas.reference           = vehicle.reference_area
    wing.spans.projected           = 36.  * Units.feet + 1. * Units.inches
    wing.sweeps.quarter_chord      = 0.*Units.degrees

    wing.thickness_to_chord        = 0.12   
    wing.chords.root               = 66. * Units.inches
    wing.chords.tip                = 45. * Units.inches
    wing.chords.mean_aerodynamic   = 58. * Units.inches # Guess
    wing.taper                     = wing.chords.root/wing.chords.tip

    wing.aspect_ratio              = wing.spans.projected**2. / wing.areas.reference

    wing.twists.root               = 3.0 * Units.degrees
    wing.twists.tip                = 1.5 * Units.degrees

    wing.origin                    = [80.* Units.inches,0,0] 
    wing.aerodynamic_center        = [22.* Units.inches,0,0]

    wing.vertical                  = False
    wing.symmetric                 = True
    wing.high_lift                 = True

    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    vehicle.append_component(wing)

    fuselage                                    = SUAVE.Components.Fuselages.Fuselage()
    fuselage.number_coach_seats                 = 4.       
    fuselage.tag                                = 'fuselage'    
    fuselage.differential_pressure              = 8*Units.psi                    # Maximum differential pressure
    fuselage.width                              = 42.         * Units.inches     # Width of the fuselage
    fuselage.heights.maximum                    = 62. * Units.inches    # Height of the fuselage
    fuselage.lengths.total                      = 326.         * Units.inches            # Length of the fuselage
    fuselage.lengths.empennage                  = 161. * Units.inches  
    fuselage.lengths.cabin                      = 105. * Units.inches

    fuselage.lengths.structure                  = fuselage.lengths.total-fuselage.lengths.empennage 

    fuselage.mass_properties.volume             = .4*fuselage.lengths.total*(np.pi/4.)*(fuselage.heights.maximum**2.) #try this as approximation
    fuselage.mass_properties.internal_volume    = .3*fuselage.lengths.total*(np.pi/4.)*(fuselage.heights.maximum**2.)
    fuselage.areas.wetted                       = 30000. * Units.inches**2.
    
    #not used for weights calculation, but keep for potential later use
    fuselage.seats_abreast                      = 2.
    fuselage.fineness.nose                      = 1.6
    fuselage.fineness.tail                      = 2.
    fuselage.lengths.nose                       = 60.  * Units.inches
    fuselage.heights.at_quarter_length          = 62. * Units.inches
    fuselage.heights.at_three_quarters_length   = 62. * Units.inches
    fuselage.heights.at_wing_root_quarter_chord = 23. * Units.inches
    fuselage.areas.front_projected              = fuselage.width* fuselage.heights.maximum
    fuselage.effective_diameter                 = 50. * Units.inches
    vehicle.append_component(fuselage)

    #horizontal tail
    wing                          = SUAVE.Components.Wings.Wing()
    wing.tag                      = 'horizontal_stabilizer'  

    wing.sweeps.quarter_chord     = 0.0 * Units.deg
    wing.areas.reference          = 5800. * Units.inches**2  # Area of the horizontal tail
    wing.spans.projected          = 136.  * Units.inches

    wing.thickness_to_chord       = 0.12                     # Thickness-to-chord ratio of the horizontal tail
    wing.chords.root              = 55. * Units.inches
    wing.chords.tip               = 30. * Units.inches
    wing.chords.mean_aerodynamic  = 43. * Units.inches # Guess

    wing.twists.root              = 0.0 * Units.degrees
    wing.twists.tip               = 0.0 * Units.degrees

    wing.taper                   = wing.chords.root/wing.chords.tip
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference

    wing.origin                  = [246.* Units.inches,0,0] 
    wing.aerodynamic_center      = [20.* Units.inches,0,0]
    wing.vertical                = False
    wing.symmetric               = True
    wing.high_lift               = False
    wing.dynamic_pressure_ratio  = 0.9
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)

    # Location of aerodynamic center from origin of the horizontal tail
    vehicle.append_component(wing)    
   
    #vertical stabilizer
    wing                         = SUAVE.Components.Wings.Wing()
    wing.tag                     = 'vertical_stabilizer'    
    wing.areas.reference         = 3500. * Units.inches**2   # Area of the vertical tail
    wing.spans.projected         = 73.   * Units.inches
    wing.sweeps.quarter_chord    = 25.     * Units.deg        # Sweep of the vertical tail

    wing.thickness_to_chord      = 0.12                      # Thickness-to-chord ratio of the vertical tail
    wing.chords.root             = 66. * Units.inches
    wing.chords.tip              = 27. * Units.inches
    wing.chords.mean_aerodynamic = 48. * Units.inches # Guess

    wing.twists.root             = 0.0 * Units.degrees
    wing.twists.tip              = 0.0 * Units.degrees

    wing.taper                   = wing.chords.root/wing.chords.tip
    wing.aspect_ratio            = wing.spans.projected**2. / wing.areas.reference
    wing.origin                  = [237.* Units.inches,0,0] 
    wing.aerodynamic_center      = [20.* Units.inches,0,0] 

    wing.t_tail                  = "false"                   # Set to "true" for a T-tail
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)

    vehicle.append_component(wing)   

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
    prop.origin              = [[2.,2.5,0.]] 
    net.propeller            = prop

    # add the network to the vehicle
    vehicle.append_component(net)  
    
    #Landing Gear
    landing_gear           = SUAVE.Components.Landing_Gear.Landing_Gear()
    main_gear              = SUAVE.Components.Landing_Gear.Main_Landing_Gear()
    nose_gear              = SUAVE.Components.Landing_Gear.Nose_Landing_Gear()
    main_gear.strut_length = 12. * Units.inches #guess based on picture
    nose_gear.strut_length = 6. * Units.inches 

    landing_gear.main      = main_gear
    landing_gear.nose      = nose_gear

    #add to vehicle
    vehicle.landing_gear   = landing_gear

    #find uninstalled avionics weight
    Wuav                                                     = 2. * Units.lbs
    avionics                                                 = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.mass_properties.uninstalled                     = Wuav
    vehicle.avionics                                         = avionics
    #fuel                                                     = SUAVE.Components.Physical_Component()
    #fuel.origin                                              = wing.origin
    #fuel.mass_properties.center_of_gravity                   = wing.mass_properties.center_of_gravity
    #fuel.mass_properties.mass                                = vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel

    '''
    #find zero_fuel_center_of_gravity
    cg                   =vehicle.mass_properties.center_of_gravity
    MTOW                 =vehicle.mass_properties.max_takeoff
    fuel_cg              =fuel.origin+fuel.mass_properties.center_of_gravity
    fuel_mass            =fuel.mass_properties.mass
    print 'cg = ', cg
    print 'fuel_cg = ', fuel_cg
    print 'MTOW = ', MTOW
    print 'fuel_mass = ', fuel_mass
    print 'fuel_cg*fuel_mass = ', fuel_cg*fuel_mass
    print 'cg*MTOW = ', cg*MTOW
    sum_moments_less_fuel=(cg*MTOW-fuel_cg*fuel_mass)
    vehicle.fuel = fuel
    vehicle.mass_properties.zero_fuel_center_of_gravity = sum_moments_less_fuel/vehicle.mass_properties.max_zero_fuel
    '''   
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
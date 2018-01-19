# Cessna_172.py
#
# Created:  Feb 2017, M. Vegh (modified from data originally in cmalpha/cmalpha.py)
# Modified: 

""" setup file for the Beech 99 aircraft, current values only used to test stability cmalpha
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x

def vehicle_setup():
    vehicle = SUAVE.Vehicle()
    GTOW    = 2200. * Units.pounds     
    vehicle.mass_properties.max_takeoff  = GTOW #from Wikipedia
    vehicle.mass_properties.empty        = 2515*Units.kg
    vehicle.mass_properties.max_zero_fuel= vehicle.mass_properties.max_takeoff-vehicle.mass_properties.empty+15.*225*Units.lbs #15 passenger ac
    vehicle.envelope.ultimate_load       = 3.8    
    vehicle.envelope.limit_load          = 1.           
    #define fuel weight needed to size fuel system
    fuel                                                = SUAVE.Attributes.Propellants.Aviation_Gasoline()
    fuel.mass_properties                                = SUAVE.Components.Mass_Properties()
    fuel.mass_properties.mass                           = 252 *Units.lbs
    fuel.number_of_tanks                                = 1.
    fuel.internal_volume                                = fuel.mass_properties.mass/fuel.density #all of the fuel volume is internal
    vehicle.fuel                                        = fuel
    
    
    propulsors = SUAVE.Components.Propulsors.Propulsor() #use weights for the IC engine  
    propulsors.tag = 'internal_combustion'
    propulsors.rated_power = 110 *Units.kW # engine correlation is really off
    #propulsors.rated_power = 100 *Units.kW
    propulsors.number_of_engines    = 1.
    vehicle.append_component(propulsors)
    
    #Build an dsize the turbofan to get sls sthrust
    cruise_speed       = 140 *Units['mph']
    altitude           = 13500 * Units.ft
    
    atmo               = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    freestream         = atmo.compute_values (0.)
    freestream0        = atmo.compute_values (altitude)
    
    mach_number        = (cruise_speed/freestream.speed_of_sound)[0][0]
    vehicle.passengers                                  = 4.  #including pilot                           # Number of passengers
    vehicle.mass_properties.cargo                       = 0.  * Units.kilogram            # Mass of cargo
    vehicle.has_air_conditioner                         = 0
    vehicle.reference_area                              = 16.2  * Units.meter**2  # Wing gross area in square meters
    vehicle.design_dynamic_pressure                     = ( .5 *freestream0.density*(cruise_speed*cruise_speed))[0][0]
    vehicle.design_mach_number                          =  mach_number
    
    
    #main wing
    wing                          = SUAVE.Components.Wings.Wing()
    wing.tag                      = 'main_wing'
    wing.areas.reference          = 175      *(Units.ft**2)
    wing.aspect_ratio             = 7.44
    wing.taper                    = 0.672                        # Taper ratio
    wing.thickness_to_chord       = 0.13                         # Thickness-to-chord ratio
    wing.sweeps.quarter_chord     = 0.       * Units.rad         # sweep angle in degrees
    wing.origin                   = [9.*Units.ft,0,0]            # Location of main wing from origin of the vehicle
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    vehicle.append_component(wing)
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'    
    fuselage.differential_pressure           = 8*Units.psi     # Maximum differential pressure
    fuselage.width                           = 40.         * Units.inches     # Width of the fuselage
    fuselage.heights.maximum                 = 75.         * Units.inches     # Height of the fuselage
    fuselage.lengths.total                   = (27+2./12.) * Units.feet     # Length of the fuselage
    fuselage.lengths.empennage               = 9           * Units.feet   
    fuselage.lengths.structure               = fuselage.lengths.total-fuselage.lengths.empennage 
    fuselage.lengths.cabin                   = 140         *Units.inches
    fuselage.mass_properties.volume          = .4*fuselage.lengths.total*(np.pi/4.)*(fuselage.heights.maximum**2.) #try this as approximation
    fuselage.mass_properties.internal_volume = .3*fuselage.lengths.total*(np.pi/4.)*(fuselage.heights.maximum**2.)
    fuselage.areas.wetted                    = fuselage.lengths.total*(np.pi/4.)*(fuselage.heights.maximum **2)
   
    fuselage.number_coach_seats       = 4.       
    vehicle.append_component(fuselage)
   
    #horizontal tail
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'    
    wing.areas.reference          = 34.6    * Units.feet**2 # Area of the horizontal tail
    wing.aspect_ratio             = 3.71
    wing.taper                    = .7
    wing.sweeps.quarter_chord     = 0.     * Units.deg       # Sweep of the horizontal tail
    wing.thickness_to_chord       = 0.13                      # Thickness-to-chord ratio of the horizontal tail
    wing.origin                    = [21.*Units.ft,0,0]                # Location of horizontal tail from origin of the vehicle

    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
                       # Location of aerodynamic center from origin of the horizontal tail
    vehicle.append_component(wing)    
   
    
    
    #vertical stabilizer
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    wing.areas.reference      = 18.4    * Units.feet**2   # Area of the vertical tail
    wing.aspect_ratio         = 1.96
    wing.taper                = .8
    wing.thickness_to_chord   = 0.13                      # Thickness-to-chord ratio of the vertical tail
    wing.sweeps.quarter_chord = 0.     * Units.deg       # Sweep of the vertical tail
    wing.t_tail               = "false"                    # Set to "yes" for a T-tail
    SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform(wing)
    
    vehicle.append_component(wing)   
    
    
    #Landing Gear
    landing_gear = SUAVE.Components.Landing_Gear.Landing_Gear()
    main_gear = SUAVE.Components.Landing_Gear.Main_Landing_Gear()
    nose_gear = SUAVE.Components.Landing_Gear.Nose_Landing_Gear()
    main_gear.strut_length = 12. * Units.inches #guess based on picture
    nose_gear.strut_length = 6. * Units.inches 
    
    landing_gear.main = main_gear
    landing_gear.nose = nose_gear
    
    #add to vehicle
    vehicle.landing_gear = landing_gear
    
    #find uninstalled avionics weight
    Wuav                                 = 2 * Units.lbs
    avionics                             = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.mass_properties.uninstalled = Wuav
    vehicle.avionics                     = avionics
    fuel                                                     =SUAVE.Components.Physical_Component()
    fuel.origin                                              =wing.origin
    fuel.mass_properties.center_of_gravity                   =wing.mass_properties.center_of_gravity
    fuel.mass_properties.mass                                =vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel
   
    
    
    
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
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    
    configs.append(config)
    
    #note: takeoff and landing configurations taken from 737 - someone should update
    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'takeoff'
    
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    
    config.V2_VS_ratio = 1.21
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    
    config.wings['main_wing'].flaps_angle = 30. * Units.deg
    config.wings['main_wing'].slats_angle = 25. * Units.deg

    config.Vref_VS_ratio = 1.23
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    
    
    # done!
    return configs
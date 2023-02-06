# Beech99.py
#
# Created:  Feb 2017, M. Vegh (modified from data originally in cmalpha/cmalpha.py)
# Modified: 

""" setup file for the Beech 99 aircraft, current values only used to test stability cmalpha
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import MARC
from MARC.Core import Units
from MARC.Core import (
    Data, Container,
)
from MARC.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
from MARC.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from MARC.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x

def vehicle_setup():
    vehicle = MARC.Vehicle()
    vehicle.mass_properties.max_takeoff      = 4727*Units.kg #from Wikipedia
    vehicle.mass_properties.empty            = 2515*Units.kg
    vehicle.mass_properties.max_zero_fuel    =vehicle.mass_properties.max_takeoff-vehicle.mass_properties.empty+15.*225*Units.lbs #15 passenger ac
    
    wing = MARC.Components.Wings.Wing()
    wing.tag                                 = 'main_wing'
    wing.areas.reference                     = 280.0 * Units.feet**2
    wing.spans.projected                     = 46.0  * Units.feet
    wing.chords.mean_aerodynamic             = 6.5 * Units.feet
    wing.chords.root                         = 7.9 * Units.feet
    wing.sweeps.leading_edge                 = 4.0   * Units.deg # Same as the quarter chord sweep (ignore why EMB)
    wing.sweeps.quarter_chord                = 4.0   * Units.deg # Leading edge
    wing.taper                               = 0.47
    wing.aspect_ratio                        = wing.spans.projected**2/wing.areas.reference
    wing.symmetric                           = True
    wing.vertical                            = False
    wing.origin                              = [[15.* Units.feet  ,0,0]]
    wing.aerodynamic_center                  = np.array([trapezoid_ac_x(wing), 0. , 0. ])
    wing.dynamic_pressure_ratio              = 1.0
    wing.ep_alpha                            = 0.0
    span_location_mac                        = compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    mac_le_offset                            =.8*np.sin(wing.sweeps.leading_edge)*span_location_mac  #assume that 80% of the chord difference is from leading edge sweep
    wing.mass_properties.center_of_gravity[0][0]=.3*wing.chords.mean_aerodynamic+mac_le_offset
    
   
    Mach                                  = np.array([0.152])
    reference                             = MARC.Core.Container()
    conditions                            = Data()
    conditions.lift_curve_slope           = datcom(wing,Mach)    
    conditions.weights                    =Data()
    conditions.weights.total_mass         =np.array([[vehicle.mass_properties.max_takeoff]])    
    wing.CL_alpha                         = conditions.lift_curve_slope
    vehicle.reference_area                = wing.areas.reference
                                          
    # control surfaces -------------------------------------------
    flap                       = MARC.Components.Wings.Control_Surfaces.Flap() 
    flap.tag                   = 'flap' 
    flap.span_fraction_start   = 0.15    # not correct, only placeholder
    flap.span_fraction_end     = 0.324   # not correct, only placeholder 
    flap.deflection            = 1.0 * Units.deg
    flap.chord_fraction        = 0.19    # not correct, only placeholder
    wing.append_control_surface(flap)    
    
    slat                       = MARC.Components.Wings.Control_Surfaces.Slat() 
    slat.tag                   = 'slat' 
    slat.span_fraction_start   = 0.324  # not correct, only placeholder 
    slat.span_fraction_end     = 0.963  # not correct, only placeholder   
    slat.deflection            = 1.0 * Units.deg
    slat.chord_fraction        = 0.1    # not correct, only placeholder 
    wing.append_control_surface(slat)  
    
    vehicle.append_component(wing)
    
    main_wing_CLa = wing.CL_alpha
    main_wing_ar  = wing.aspect_ratio
    
    wing = MARC.Components.Wings.Wing()
    wing.tag                      = 'horizontal_stabilizer'
    wing.areas.reference          = 100.5 * Units.feet**2
    wing.spans.projected          = 22.5  * Units.feet
    wing.sweeps.leading_edge      = 21.0   * Units.deg # Same as the quarter chord sweep (ignore why EMB)
    wing.sweeps.quarter_chord     = 21.0   * Units.deg # leading edge
    wing.taper                    = 3.1/6.17
    wing.aspect_ratio             = wing.spans.projected**2/wing.areas.reference
    wing.origin                   = [[36.3* Units.feet,0,0]]
    wing.symmetric                = True
    wing.vertical                 = False
    wing.dynamic_pressure_ratio   = 0.95
    wing.ep_alpha                 = 2.0*main_wing_CLa/np.pi/main_wing_ar
    wing.aerodynamic_center       = np.array([trapezoid_ac_x(wing), 0.0, 0.0])
    wing.CL_alpha                 = datcom(wing,Mach)
    vehicle.append_component(wing)
    
    fuselage = MARC.Components.Fuselages.Fuselage()
    fuselage.tag                  = 'fuselage'
    fuselage.x_root_quarter_chord = 5.4 * Units.feet
    fuselage.lengths.total        = 44.0  * Units.feet
    fuselage.width                = 5.4   * Units.feet 
    vehicle.append_component(fuselage)
    
    vehicle.mass_properties.center_of_gravity = np.array([[17.2,0,0]]) * Units.feet  
    fuel                                      = MARC.Components.Physical_Component()
    fuel.origin                               = wing.origin
    fuel.mass_properties.center_of_gravity    = wing.mass_properties.center_of_gravity
    fuel.mass_properties.mass                 = vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel
    
    #find zero_fuel_center_of_gravity
    cg                     = vehicle.mass_properties.center_of_gravity
    MTOW                   = vehicle.mass_properties.max_takeoff
    fuel_cg                = fuel.origin+fuel.mass_properties.center_of_gravity
    fuel_mass              = fuel.mass_properties.mass 
    sum_moments_less_fuel  = (cg*MTOW-fuel_cg*fuel_mass)
    vehicle.fuel           =  fuel
    
    vehicle.mass_properties.zero_fuel_center_of_gravity = sum_moments_less_fuel/vehicle.mass_properties.max_zero_fuel
    
    return vehicle
  
def configs_setup(vehicle):
     # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    configs = MARC.Components.Configs.Config.Container()
    base_config = MARC.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------ 
    config = MARC.Components.Configs.Config(base_config)
    config.tag = 'cruise' 
    configs.append(config)
                   
    #note: takeoff and landing configurations taken from 737 - someone should update
    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------ 
    config = MARC.Components.Configs.Config(base_config)
    config.tag = 'takeoff'    
    config.wings['main_wing'].control_surfaces.flap.deflection = 20. * Units.deg
    config.wings['main_wing'].control_surfaces.slat.deflection = 25. * Units.deg 
    config.V2_VS_ratio = 1.21
    config.maximum_lift_coefficient = 2. 
    configs.append(config)
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------  
    config = MARC.Components.Configs.Config(base_config)
    config.tag = 'landing'    
    config.wings['main_wing'].flaps_angle = 30. * Units.deg
    config.wings['main_wing'].slats_angle = 25. * Units.deg
    config.Vref_VS_ratio = 1.23
    config.maximum_lift_coefficient = 2.
    configs.append(config)
    
    
    # done!
    return configs
# test_cmalpha.py
# Tim Momose, April 2014
# Reference: Aircraft Dynamics: from Modeling to Simulation, by M. R. Napolitano

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_mac import trapezoid_mac
#from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.convert_sweep import convert_sweep
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
#from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approsimations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Tube_Wing.taw_cmalpha import taw_cmalpha
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length

from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container,
)
def main():
    #Parameters Required
    #Using values for a Boeing 747-200  
    vehicle = SUAVE.Vehicle()
    #print vehicle
    vehicle.mass_properties.max_zero_fuel=238780*Units.kg
    vehicle.mass_properties.max_takeoff  =785000.*Units.lbs
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    wing.areas.reference           = 5500.0 * Units.feet**2
    wing.spans.projected           = 196.0  * Units.feet
    wing.chords.mean_aerodynamic   = 27.3 * Units.feet
    wing.chords.root               = 42.9 * Units.feet  #54.5ft
    wing.sweeps.quarter_chord      = 42.0   * Units.deg  # Leading edge
    wing.sweeps.leading_edge       = 42.0   * Units.deg  # Same as the quarter chord sweep (ignore why EMB)
    wing.taper          = 14.7/42.9  #14.7/54.5
    wing.aspect_ratio   = wing.spans.projected**2/wing.areas.reference
    wing.symmetric      = True
    wing.vertical       = False
    wing.origin         = np.array([58.6,0,0]) * Units.feet  
    wing.aerodynamic_center     = np.array([112.2*Units.feet,0.,0.])-wing.origin#16.16 * Units.meters,0.,0,])
    wing.dynamic_pressure_ratio = 1.0
    wing.ep_alpha               = 0.0
    
    span_location_mac                        =compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    mac_le_offset                            =.8*np.sin(wing.sweeps.leading_edge)*span_location_mac  #assume that 80% of the chord difference is from leading edge sweep
    wing.mass_properties.center_of_gravity[0]=.3*wing.chords.mean_aerodynamic+mac_le_offset
    
    
    Mach                         = np.array([0.198])
    conditions                   = Data()
    conditions.weights           = Data()
    conditions.lift_curve_slope  = datcom(wing,Mach)
    conditions.weights.total_mass=np.array([[vehicle.mass_properties.max_takeoff]]) 
   
    wing.CL_alpha                = conditions.lift_curve_slope
    vehicle.reference_area       = wing.areas.reference
    vehicle.append_component(wing)
    
    main_wing_CLa = wing.CL_alpha
    main_wing_ar  = wing.aspect_ratio
    
    wing                     = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'
    wing.areas.reference      = 1490.55* Units.feet**2
    wing.spans.projected      = 71.6   * Units.feet
    wing.sweeps.quarter_chord = 44.0   * Units.deg # leading edge
    wing.sweeps.leading_edge  = 44.0   * Units.deg # Same as the quarter chord sweep (ignore why EMB)
    wing.taper                = 7.5/32.6
    wing.aspect_ratio         = wing.spans.projected**2/wing.areas.reference
    wing.origin               = np.array([187.0,0,0])  * Units.feet
    wing.symmetric            = True
    wing.vertical             = False
    wing.dynamic_pressure_ratio = 0.95
    wing.ep_alpha             = 2.0*main_wing_CLa/np.pi/main_wing_ar    
    wing.aerodynamic_center   = [trapezoid_ac_x(wing), 0.0, 0.0]
    wing.CL_alpha             = datcom(wing,Mach)
    vehicle.append_component(wing)
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.x_root_quarter_chord = 77.0 * Units.feet
    fuselage.lengths.total     = 229.7  * Units.feet
    fuselage.width      = 20.9   * Units.feet 
    vehicle.append_component(fuselage)
    vehicle.mass_properties.center_of_gravity=np.array([112.2,0,0]) * Units.feet  
    
    
    
 
    #configuration.mass_properties.zero_fuel_center_of_gravity=np.array([76.5,0,0])*Units.feet #just put a number here that got the expected value output; may want to change
    fuel                                                     =SUAVE.Components.Physical_Component()
    fuel.origin                                              =wing.origin
    fuel.mass_properties.center_of_gravity                   =wing.mass_properties.center_of_gravity
    fuel.mass_properties.mass                                =vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel
   
    
    #find zero_fuel_center_of_gravity
    cg                   =vehicle.mass_properties.center_of_gravity
    MTOW                 =vehicle.mass_properties.max_takeoff
    fuel_cg              =fuel.origin+fuel.mass_properties.center_of_gravity
    fuel_mass            =fuel.mass_properties.mass
    
    
    sum_moments_less_fuel=(cg*MTOW-fuel_cg*fuel_mass)
    
    
    #now define configuration for calculation
    configuration = Data()
    configuration.mass_properties                            = Data()
    configuration.mass_properties.center_of_gravity          = vehicle.mass_properties.center_of_gravity
    configuration.mass_properties.max_zero_fuel              =vehicle.mass_properties.max_zero_fuel
    configuration.fuel                                       =fuel
    
    configuration.mass_properties.zero_fuel_center_of_gravity=sum_moments_less_fuel/vehicle.mass_properties.max_zero_fuel
  
    
    #print configuration
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configuration)
    expected =-1.56222373 #Should be -1.45
    error = Data()
    error.cm_a_747 = (cm_a - expected)/expected
    
    
    #Parameters Required
    #Using values for a Beech 99 
    
    vehicle = SUAVE.Vehicle()
    vehicle.mass_properties.max_takeoff  =4727*Units.kg #from Wikipedia
    vehicle.mass_properties.empty        =2515*Units.kg
    vehicle.mass_properties.max_zero_fuel=vehicle.mass_properties.max_takeoff-vehicle.mass_properties.empty+15.*225*Units.lbs #15 passenger ac
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    wing.areas.reference           = 280.0 * Units.feet**2
    wing.spans.projected           = 46.0  * Units.feet
    wing.chords.mean_aerodynamic   = 6.5 * Units.feet
    wing.chords.root               = 7.9 * Units.feet
    wing.sweeps.leading_edge       = 4.0   * Units.deg # Same as the quarter chord sweep (ignore why EMB)
    wing.sweeps.quarter_chord      = 4.0   * Units.deg # Leading edge
    wing.taper                     = 0.47
    wing.aspect_ratio              = wing.spans.projected**2/wing.areas.reference
    wing.symmetric                 = True
    wing.vertical                  = False
    wing.origin                    = np.array([15.,0,0]) * Units.feet  
    wing.aerodynamic_center        = np.array([trapezoid_ac_x(wing), 0. , 0. ])
    wing.dynamic_pressure_ratio    = 1.0
    wing.ep_alpha                  = 0.0
    span_location_mac                        =compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    mac_le_offset                            =.8*np.sin(wing.sweeps.leading_edge)*span_location_mac  #assume that 80% of the chord difference is from leading edge sweep
    wing.mass_properties.center_of_gravity[0]=.3*wing.chords.mean_aerodynamic+mac_le_offset
    
    
    
    
    
    Mach = np.array([0.152])
    reference = SUAVE.Core.Container()
    conditions = Data()
    conditions.lift_curve_slope = datcom(wing,Mach)
    
    conditions.weights=Data()
    conditions.weights.total_mass=np.array([[vehicle.mass_properties.max_takeoff]]) 
   
    wing.CL_alpha               = conditions.lift_curve_slope
    vehicle.reference_area      = wing.areas.reference
    vehicle.append_component(wing)
    
    main_wing_CLa = wing.CL_alpha
    main_wing_ar  = wing.aspect_ratio
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag                      = 'horizontal_stabilizer'
    wing.areas.reference          = 100.5 * Units.feet**2
    wing.spans.projected          = 22.5  * Units.feet
    wing.sweeps.leading_edge      = 21.0   * Units.deg # Same as the quarter chord sweep (ignore why EMB)
    wing.sweeps.quarter_chord     = 21.0   * Units.deg # leading edge
    wing.taper                    = 3.1/6.17
    wing.aspect_ratio             = wing.spans.projected**2/wing.areas.reference
    wing.origin                   = np.array([36.3,0,0])  * Units.feet
    wing.symmetric                = True
    wing.vertical                 = False
    wing.dynamic_pressure_ratio   = 0.95
    wing.ep_alpha                 = 2.0*main_wing_CLa/np.pi/main_wing_ar
    wing.aerodynamic_center       = np.array([trapezoid_ac_x(wing), 0.0, 0.0])
    wing.CL_alpha                 = datcom(wing,Mach)
    vehicle.append_component(wing)
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag                  = 'fuselage'
    fuselage.x_root_quarter_chord = 5.4 * Units.feet
    fuselage.lengths.total        = 44.0  * Units.feet
    fuselage.width                = 5.4   * Units.feet 
    vehicle.append_component(fuselage)
    
    vehicle.mass_properties.center_of_gravity = np.array([17.2,0,0]) * Units.feet   
    
    
    fuel.origin                                              =wing.origin
    fuel.mass_properties.center_of_gravity                   =wing.mass_properties.center_of_gravity
    fuel.mass_properties.mass                                =vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel
   
    
    
    #find zero_fuel_center_of_gravity
    cg                   =vehicle.mass_properties.center_of_gravity
    MTOW                 =vehicle.mass_properties.max_takeoff
    fuel_cg              =fuel.origin+fuel.mass_properties.center_of_gravity
    fuel_mass            =fuel.mass_properties.mass

    
    sum_moments_less_fuel=(cg*MTOW-fuel_cg*fuel_mass)
    
    
    #now define configuration for calculation
    configuration = Data()
    configuration.mass_properties                            = Data()
    configuration.mass_properties.center_of_gravity          = vehicle.mass_properties.center_of_gravity
    configuration.mass_properties.max_zero_fuel              = vehicle.mass_properties.max_zero_fuel
    configuration.fuel                                       =fuel
    
    configuration.mass_properties.zero_fuel_center_of_gravity=sum_moments_less_fuel/vehicle.mass_properties.max_zero_fuel
    
    
    #Method Test   
    #print configuration
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configuration)
    expected = -2.48843437 #Should be -2.08
    error.cm_a_beech_99 = (cm_a - expected)/expected   
    
    
    #Parameters Required
    #Using values for an SIAI Marchetti S-211
    
    vehicle = SUAVE.Vehicle()
    vehicle.mass_properties.max_takeoff  =2750*Units.kg #from Wikipedia
    vehicle.mass_properties.empty        =1850*Units.kg
    vehicle.mass_properties.max_zero_fuel=vehicle.mass_properties.max_takeoff-vehicle.mass_properties.empty+2.*225*Units.lbs #2 passenger ac
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    wing.areas.reference           = 136.0 * Units.feet**2
    wing.spans.projected           = 26.3  * Units.feet
    wing.chords.mean_aerodynamic   = 5.4   * Units.feet
    wing.chords.root               = 7.03  * Units.feet
    wing.chords.tip                = 3.1   * Units.feet
    wing.sweeps.quarter_chord      = 19.5  * Units.deg # Leading edge
    wing.sweeps.leading_edge       = 19.5  * Units.deg # Same as the quarter chord sweep (ignore why EMB)
    wing.taper          = 3.1/7.03
    wing.aspect_ratio   = wing.spans.projected**2/wing.areas.reference
    wing.symmetric      = True
    wing.vertical       = False
    wing.origin         = np.array([13.5,0,0]) * Units.feet  
    wing.aerodynamic_center  = np.array([trapezoid_ac_x(wing),0.,0.])#16.6, 0. , 0. ]) * Units.feet - wing.origin
    wing.dynamic_pressure_ratio = 1.0
    wing.ep_alpha       = 0.0
    
    span_location_mac                        =compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    mac_le_offset                            =.8*np.sin(wing.sweeps.leading_edge)*span_location_mac  #assume that 80% of the chord difference is from leading edge sweep
    wing.mass_properties.center_of_gravity[0]=.3*wing.chords.mean_aerodynamic+mac_le_offset
    
       
    
    Mach = np.array([0.111])
    conditions = Data()
    conditions.lift_curve_slope = datcom(wing,Mach)
    conditions.weights=Data()
    conditions.weights.total_mass=np.array([[vehicle.mass_properties.max_takeoff]]) 
   
    
    
    
    wing.CL_alpha               = conditions.lift_curve_slope
    vehicle.reference_area      = wing.areas.reference
    vehicle.append_component(wing)
    
    main_wing_CLa = wing.CL_alpha
    main_wing_ar  = wing.aspect_ratio

    wing = SUAVE.Components.Wings.Wing()
    wing.tag                 = 'horizontal_stabilizer'
    wing.areas.reference     = 36.46 * Units.feet**2
    wing.spans.projected     = 13.3   * Units.feet
    wing.sweeps.quarter_chord= 18.5  * Units.deg  # leading edge
    wing.sweeps.leading_edge = 18.5  * Units.deg  # Same as the quarter chord sweep (ignore why EMB)
    wing.taper               = 1.6/3.88
    wing.aspect_ratio        = wing.spans.projected**2/wing.areas.reference
    wing.origin              = np.array([26.07,0.,0.]) * Units.feet
    wing.symmetric           = True
    wing.vertical            = False
    wing.dynamic_pressure_ratio = 0.9
    wing.ep_alpha            = 2.0*main_wing_CLa/np.pi/main_wing_ar
    wing.aerodynamic_center  = np.array([trapezoid_ac_x(wing), 0.0, 0.0])
    wing.CL_alpha            = datcom(wing,Mach)
    
    span_location_mac                        =compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    mac_le_offset                            =.8*np.sin(wing.sweeps.leading_edge)*span_location_mac  #assume that 80% of the chord difference is from leading edge sweep
    wing.mass_properties.center_of_gravity[0]=.3*wing.chords.mean_aerodynamic+mac_le_offset
    
    
    
    
    
    
    
    vehicle.append_component(wing)
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.x_root_quarter_chord = 12.67 * Units.feet
    fuselage.lengths.total        = 30.9  * Units.feet
    fuselage.width                = ((2.94+5.9)/2)   * Units.feet 
    vehicle.append_component(fuselage)
    vehicle.mass_properties.center_of_gravity = np.array([16.6,0,0]) * Units.feet    
    
    
    
    
    fuel.origin                                              =wing.origin
    fuel.mass_properties.center_of_gravity                   =wing.mass_properties.center_of_gravity
    fuel.mass_properties.mass                                =vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel
   
    
    
    #find zero_fuel_center_of_gravity
    cg                   =vehicle.mass_properties.center_of_gravity
    MTOW                 =vehicle.mass_properties.max_takeoff
    fuel_cg              =fuel.origin+fuel.mass_properties.center_of_gravity
    fuel_mass            =fuel.mass_properties.mass

    sum_moments_less_fuel=(cg*MTOW-fuel_cg*fuel_mass)
    
    
    #now define configuration for calculation
    configuration = Data()
    configuration.mass_properties                            = Data()
    configuration.mass_properties.center_of_gravity          = vehicle.mass_properties.center_of_gravity
    configuration.mass_properties.max_zero_fuel              = vehicle.mass_properties.max_zero_fuel
    configuration.fuel                                       =fuel
    
    configuration.mass_properties.zero_fuel_center_of_gravity=sum_moments_less_fuel/vehicle.mass_properties.max_zero_fuel
    
    
    
    #Method Test   
    #print configuration
    cm_a = taw_cmalpha(vehicle,Mach,conditions,configuration)
   
    expected = -0.54071741 #Should be -0.6
    error.cm_a_SIAI = (cm_a - expected)/expected
    print error
    for k,v in error.items():
        assert(np.abs(v)<0.01)
        
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()

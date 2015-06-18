# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)


def main():
    vehicle = SUAVE.Vehicle()# Create the vehicle for testing
    
    # Parameters Required
    vehicle.envelope.ultimate_load                      = 3.5                             # Ultimate load
    vehicle.mass_properties.max_takeoff                 = 79015.8 * Units.kilograms       # Maximum takeoff weight in kilograms
    vehicle.mass_properties.max_zero_fuel               = 79015.8 * 0.9 * Units.kilograms # Maximum zero fuel weight in kilograms
    vehicle.envelope.limit_load                         = 1.5                             # Limit Load
    
    turbofan = SUAVE.Components.Propulsors.TurboFanPASS()
    turbofan.tag = 'turbo_fan'    
    turbofan.number_of_engines   = 2.                              # Number of engines on the aircraft
    turbofan.design_thrust  = 200.   * Units.newton    # Define Thrust in Newtons    
    vehicle.append_component(turbofan) 

    vehicle.passengers                                  = 170.                            # Number of passengers
    vehicle.mass_properties.cargo                       = 0.  * Units.kilogram            # Mass of cargo
    vehicle.systems.control                             = "fully powered"                 # Specify fully powered, partially powered or anything else is fully aerodynamic
    vehicle.systems.accessories                         = "medium-range"                  # Specify what type of aircraft you have
    
    vehicle.reference_area                              = 124.862  * Units.meter**2  # Wing gross area in square meters
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'main_wing'
    wing.spans.projected          = 50.      * Units.meter     # Span in meters
    wing.taper                    = 0.2                        # Taper ratio
    wing.thickness_to_chord       = 0.08                       # Thickness-to-chord ratio
    wing.sweep                    = .4363323 * Units.rad       # sweep angle in degrees
    wing.chords.root              = 15.      * Units.meter     # Wing root chord length
    wing.chords.mean_aerodynamic  = 10.      * Units.meters    # Length of the mean aerodynamic chord of the wing
    wing.origin                 = [20,0,0] * Units.meters    # Location of main wing from origin of the vehicle
    wing.aerodynamic_center       = [3,0,0]  * Units.meters    # Location of aerodynamic center from origin of the main wing
    vehicle.append_component(wing)
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'    
    fuselage.areas.wetted             = 688.64    * Units.meter**2  # Fuselage wetted area 
    fuselage.differential_pressure    = 55960.5   * Units.pascal    # Maximum differential pressure
    fuselage.width                    = 4.        * Units.meter     # Width of the fuselage
    fuselage.heights.maximum          = 4.        * Units.meter     # Height of the fuselage
    fuselage.lengths.total            = 58.4      * Units.meter     # Length of the fuselage
    fuselage.number_coach_seats       = 200.       
    vehicle.append_component(fuselage)
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'horizontal_stabilizer'    
    wing.areas.reference          = 75.     * Units.meters**2 # Area of the horizontal tail
    wing.spans.projected          = 15.     * Units.meters    # Span of the horizontal tail
    wing.sweep                    = 38.     * Units.deg       # Sweep of the horizontal tail
    wing.chords.mean_aerodynamic  = 5.      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    wing.thickness_to_chord       = 0.07                      # Thickness-to-chord ratio of the horizontal tail
    wing.areas.exposed            = 199.7792                  # Exposed area of the horizontal tail
    wing.areas.wetted             = 249.724                   # Wetted area of the horizontal tail
    wing.origin                 = [45,0,0]                  # Location of horizontal tail from origin of the vehicle
    wing.aerodynamic_center       = [3,0,0]                   # Location of aerodynamic center from origin of the horizontal tail
    vehicle.append_component(wing)    
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    wing.areas.reference     = 60.     * Units.meters**2 # Area of the vertical tail
    wing.spans.projected     = 15.     * Units.meters    # Span of the vertical tail
    wing.thickness_to_chord  = 0.07                      # Thickness-to-chord ratio of the vertical tail
    wing.sweep               = 40.     * Units.deg       # Sweep of the vertical tail
    wing.t_tail              = "false"                   # Set to "yes" for a T-tail
    vehicle.append_component(wing)   
    
    weight = Tube_Wing.empty(vehicle)
    
    actual = Data()
    actual.payload = 17349.9081525
    actual.pax = 15036.5870655
    actual.bag = 2313.321087
    actual.fuel = -6993.89102491
    actual.empty = 68659.7828724
    actual.wing = 27694.192985
    actual.fuselage = 11504.5186408
    actual.propulsion = 88.3696093424
    actual.landing_gear = 3160.632
    actual.systems = 16655.7076511
    actual.wt_furnish = 7466.1304102
    actual.horizontal_tail = 2191.30720639
    actual.vertical_tail = 5260.75341411
    actual.rudder = 2104.30136565    
    
    error = Data()
    error.payload = (actual.payload - weight.payload)/actual.payload
    error.pax = (actual.pax - weight.pax)/actual.pax
    error.bag = (actual.bag - weight.bag)/actual.bag
    error.fuel = (actual.fuel - weight.fuel)/actual.fuel
    error.empty = (actual.empty - weight.empty)/actual.empty
    error.wing = (actual.wing - weight.wing)/actual.wing
    error.fuselage = (actual.fuselage - weight.fuselage)/actual.fuselage
    error.propulsion = (actual.propulsion - weight.propulsion)/actual.propulsion
    error.landing_gear = (actual.landing_gear - weight.landing_gear)/actual.landing_gear
    error.systems = (actual.systems - weight.systems)/actual.systems
    error.wt_furnish = (actual.wt_furnish - weight.wt_furnish)/actual.wt_furnish
    error.horizontal_tail = (actual.horizontal_tail - weight.horizontal_tail)/actual.horizontal_tail
    error.vertical_tail = (actual.vertical_tail - weight.vertical_tail)/actual.vertical_tail
    error.rudder = (actual.rudder - weight.rudder)/actual.rudder
      
    for k,v in error.items():
        assert(np.abs(v)<0.001)    
    
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()    
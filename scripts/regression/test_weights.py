# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)


def main():
    
    vehicle = Data()
    vehicle.envelope = Data()
    vehicle.mass_properties = Data()
    vehicle.propulsors = Data()
    vehicle.propulsors['Turbo Fan'] = Data()
    vehicle.propulsors['Turbo Fan'].thrust = Data()
    vehicle.propulsors['Turbo Fan'].mass_properties = Data()
    vehicle.fuselages = Data()
    vehicle.fuselages.fuselage = Data()
    vehicle.fuselages.fuselage.areas = Data()
    vehicle.fuselages.fuselage.heights = Data()
    vehicle.fuselages.fuselage.lengths = Data()
    vehicle.fuselages.fuselage.mass_properties = Data()
    vehicle.systems = Data()
    vehicle.wings = Data()
    vehicle.wings['Main Wing'] = Data()
    vehicle.wings['Main Wing'].spans = Data()
    vehicle.wings['Main Wing'].chords = Data()
    vehicle.wings['Main Wing'].mass_properties = Data()
    vehicle.wings['Horizontal Stabilizer'] = Data()
    vehicle.wings['Horizontal Stabilizer'].areas = Data()
    vehicle.wings['Horizontal Stabilizer'].spans = Data()
    vehicle.wings['Horizontal Stabilizer'].chords = Data()
    vehicle.wings['Horizontal Stabilizer'].mass_properties = Data()
    vehicle.wings['Vertical Stabilizer'] = Data()
    vehicle.wings['Vertical Stabilizer'].areas = Data()
    vehicle.wings['Vertical Stabilizer'].spans = Data()
    vehicle.wings['Vertical Stabilizer'].mass_properties = Data()
    
    
    # Parameters Required
    vehicle.envelope.ultimate_load                      = 3.5                             # Ultimate load
    vehicle.mass_properties.max_takeoff                 = 79015.8 * Units.kilograms       # Maximum takeoff weight in kilograms
    vehicle.mass_properties.max_zero_fuel               = 79015.8 * 0.9 * Units.kilograms # Maximum zero fuel weight in kilograms
    vehicle.envelope.limit_load                         = 1.5                             # Limit Load
    vehicle.propulsors['Turbo Fan'].number_of_engines   = 2.                              # Number of engines on the aircraft
    vehicle.passengers                                  = 170.                            # Number of passengers
    vehicle.mass_properties.cargo                       = 0.  * Units.kilogram            # Mass of cargo
    vehicle.fuselages.fuselage.number_coach_seats          = 200.                            # Number of seats on aircraft
    vehicle.systems.control                             = "fully powered"                 # Specify fully powered, partially powered or anything else is fully aerodynamic
    vehicle.systems.accessories                         = "medium-range"                  # Specify what type of aircraft you have
    
    vehicle.reference_area                              = 124.862  * Units.meter**2  # Wing gross area in square meters
    vehicle.wings['Main Wing'].spans.projected          = 50.      * Units.meter     # Span in meters
    vehicle.wings['Main Wing'].taper                    = 0.2                        # Taper ratio
    vehicle.wings['Main Wing'].thickness_to_chord       = 0.08                       # Thickness-to-chord ratio
    vehicle.wings['Main Wing'].sweep                    = .4363323 * Units.rad       # sweep angle in degrees
    vehicle.wings['Main Wing'].chords.root              = 15.      * Units.meter     # Wing root chord length
    vehicle.wings['Main Wing'].chords.mean_aerodynamic  = 10.      * Units.meters    # Length of the mean aerodynamic chord of the wing
    vehicle.wings['Main Wing'].position                 = [20,0,0] * Units.meters    # Location of main wing from origin of the vehicle
    vehicle.wings['Main Wing'].aerodynamic_center       = [3,0,0]  * Units.meters    # Location of aerodynamic center from origin of the main wing
    
    
    vehicle.fuselages.fuselage.areas.wetted             = 688.64    * Units.meter**2  # Fuselage wetted area 
    vehicle.fuselages.fuselage.differential_pressure    = 55960.5   * Units.pascal    # Maximum differential pressure
    vehicle.fuselages.fuselage.width                    = 4.        * Units.meter     # Width of the fuselage
    vehicle.fuselages.fuselage.heights.maximum          = 4.        * Units.meter     # Height of the fuselage
    vehicle.fuselages.fuselage.lengths.total            = 58.4      * Units.meter     # Length of the fuselage
    
    vehicle.propulsors['Turbo Fan'].thrust.design  = 1000.   * Units.newton    # Define Thrust in Newtons
    
    vehicle.wings['Horizontal Stabilizer'].areas.reference          = 75.     * Units.meters**2 # Area of the horizontal tail
    vehicle.wings['Horizontal Stabilizer'].spans.projected          = 15.     * Units.meters    # Span of the horizontal tail
    vehicle.wings['Horizontal Stabilizer'].sweep                    = 38.     * Units.deg       # Sweep of the horizontal tail
    vehicle.wings['Horizontal Stabilizer'].chords.mean_aerodynamic  = 5.      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    vehicle.wings['Horizontal Stabilizer'].thickness_to_chord       = 0.07                      # Thickness-to-chord ratio of the horizontal tail
    vehicle.wings['Horizontal Stabilizer'].areas.exposed            = 199.7792                  # Exposed area of the horizontal tail
    vehicle.wings['Horizontal Stabilizer'].areas.wetted             = 249.724                   # Wetted area of the horizontal tail
    vehicle.wings['Horizontal Stabilizer'].position                 = [45,0,0]                  # Location of horizontal tail from origin of the vehicle
    vehicle.wings['Horizontal Stabilizer'].aerodynamic_center       = [3,0,0]                   # Location of aerodynamic center from origin of the horizontal tail
    
    vehicle.wings['Vertical Stabilizer'].areas.reference     = 60.     * Units.meters**2 # Area of the vertical tail
    vehicle.wings['Vertical Stabilizer'].spans.projected     = 15.     * Units.meters    # Span of the vertical tail
    vehicle.wings['Vertical Stabilizer'].thickness_to_chord  = 0.07                      # Thickness-to-chord ratio of the vertical tail
    vehicle.wings['Vertical Stabilizer'].sweep               = 40.     * Units.deg       # Sweep of the vertical tail
    vehicle.wings['Vertical Stabilizer'].t_tail              = "false"                   # Set to "yes" for a T-tail
    
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
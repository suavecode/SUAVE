# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Attributes import Units as Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# new units style
a = 4 * Units.mm # convert into base units
b = a / Units.mm # convert out of base units
def main():
    
    vehicle = Data()
    vehicle.Envelope = Data()
    vehicle.Mass_Properties = Data()
    vehicle.Propulsors = Data()
    vehicle.Propulsors['Turbo Fan'] = Data()
    vehicle.Propulsors['Turbo Fan'].Thrust = Data()
    vehicle.Propulsors['Turbo Fan'].Mass_Properties = Data()
    vehicle.Fuselages = Data()
    vehicle.Fuselages.Fuselage = Data()
    vehicle.Fuselages.Fuselage.Areas = Data()
    vehicle.Fuselages.Fuselage.Heights = Data()
    vehicle.Fuselages.Fuselage.Lengths = Data()
    vehicle.Fuselages.Fuselage.Mass_Properties = Data()
    vehicle.Systems = Data()
    vehicle.Wings = Data()
    vehicle.Wings['Main Wing'] = Data()
    vehicle.Wings['Main Wing'].Spans = Data()
    vehicle.Wings['Main Wing'].Chords = Data()
    vehicle.Wings['Main Wing'].Mass_Properties = Data()
    vehicle.Wings['Horizontal Stabilizer'] = Data()
    vehicle.Wings['Horizontal Stabilizer'].Areas = Data()
    vehicle.Wings['Horizontal Stabilizer'].Spans = Data()
    vehicle.Wings['Horizontal Stabilizer'].Chords = Data()
    vehicle.Wings['Horizontal Stabilizer'].Mass_Properties = Data()
    vehicle.Wings['Vertical Stabilizer'] = Data()
    vehicle.Wings['Vertical Stabilizer'].Areas = Data()
    vehicle.Wings['Vertical Stabilizer'].Spans = Data()
    vehicle.Wings['Vertical Stabilizer'].Mass_Properties = Data()
    
    
    # Parameters Required
    vehicle.Envelope.ultimate_load                      = 3.5                             # Ultimate load
    vehicle.Mass_Properties.max_takeoff                 = 79015.8 * Units.kilograms       # Maximum takeoff weight in kilograms
    vehicle.Mass_Properties.max_zero_fuel               = 79015.8 * 0.9 * Units.kilograms # Maximum zero fuel weight in kilograms
    vehicle.Envelope.limit_load                         = 1.5                             # Limit Load
    vehicle.Propulsors['Turbo Fan'].number_of_engines   = 2.                              # Number of engines on the aircraft
    vehicle.passengers                                  = 170.                            # Number of passengers
    vehicle.Mass_Properties.cargo                       = 0.  * Units.kilogram            # Mass of cargo
    vehicle.Fuselages.Fuselage.num_coach_seats          = 200.                            # Number of seats on aircraft
    vehicle.Systems.control                             = "fully powered"                 # Specify fully powered, partially powered or anything else is fully aerodynamic
    vehicle.Systems.accessories                         = "medium-range"                  # Specify what type of aircraft you have
    
    vehicle.reference_area                              = 124.862  * Units.meter**2  # Wing gross area in square meters
    vehicle.Wings['Main Wing'].Spans.projected          = 50.      * Units.meter     # Span in meters
    vehicle.Wings['Main Wing'].taper                    = 0.2                        # Taper ratio
    vehicle.Wings['Main Wing'].thickness_to_chord       = 0.08                       # Thickness-to-chord ratio
    vehicle.Wings['Main Wing'].sweep                    = .4363323 * Units.rad       # sweep angle in degrees
    vehicle.Wings['Main Wing'].Chords.root              = 15.      * Units.meter     # Wing root chord length
    vehicle.Wings['Main Wing'].Chords.mean_aerodynamic  = 10.      * Units.meters    # Length of the mean aerodynamic chord of the wing
    vehicle.Wings['Main Wing'].position                 = [20,0,0] * Units.meters    # Location of main wing from origin of the vehicle
    vehicle.Wings['Main Wing'].aerodynamic_center       = [3,0,0]  * Units.meters    # Location of aerodynamic center from origin of the main wing
    
    
    vehicle.Fuselages.Fuselage.Areas.wetted             = 688.64    * Units.meter**2  # Fuselage wetted area 
    vehicle.Fuselages.Fuselage.differential_pressure    = 55960.5   * Units.pascal    # Maximum differential pressure
    vehicle.Fuselages.Fuselage.width                    = 4.        * Units.meter     # Width of the fuselage
    vehicle.Fuselages.Fuselage.Heights.maximum          = 4.        * Units.meter     # Height of the fuselage
    vehicle.Fuselages.Fuselage.Lengths.total            = 58.4      * Units.meter     # Length of the fuselage
    
    vehicle.Propulsors['Turbo Fan'].Thrust.design  = 1000.   * Units.newton    # Define Thrust in Newtons
    
    vehicle.Wings['Horizontal Stabilizer'].Areas.reference          = 75.     * Units.meters**2 # Area of the horizontal tail
    vehicle.Wings['Horizontal Stabilizer'].Spans.projected          = 15.     * Units.meters    # Span of the horizontal tail
    vehicle.Wings['Horizontal Stabilizer'].sweep                    = 38.     * Units.deg       # Sweep of the horizontal tail
    vehicle.Wings['Horizontal Stabilizer'].Chords.mean_aerodynamic  = 5.      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    vehicle.Wings['Horizontal Stabilizer'].thickness_to_chord       = 0.07                      # Thickness-to-chord ratio of the horizontal tail
    vehicle.Wings['Horizontal Stabilizer'].Areas.exposed            = 199.7792                  # Exposed area of the horizontal tail
    vehicle.Wings['Horizontal Stabilizer'].Areas.wetted             = 249.724                   # Wetted area of the horizontal tail
    vehicle.Wings['Horizontal Stabilizer'].position                 = [45,0,0]                  # Location of horizontal tail from origin of the vehicle
    vehicle.Wings['Horizontal Stabilizer'].aerodynamic_center       = [3,0,0]                   # Location of aerodynamic center from origin of the horizontal tail
    
    vehicle.Wings['Vertical Stabilizer'].Areas.reference     = 60.     * Units.meters**2 # Area of the vertical tail
    vehicle.Wings['Vertical Stabilizer'].Spans.projected     = 15.     * Units.meters    # Span of the vertical tail
    vehicle.Wings['Vertical Stabilizer'].thickness_to_chord  = 0.07                      # Thickness-to-chord ratio of the vertical tail
    vehicle.Wings['Vertical Stabilizer'].sweep               = 40.     * Units.deg       # Sweep of the vertical tail
    vehicle.Wings['Vertical Stabilizer'].t_tail              = "false"                   # Set to "yes" for a T-tail
    
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
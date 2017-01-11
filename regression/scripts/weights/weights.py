# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

def main():
    vehicle = SUAVE.Vehicle()# Create the vehicle for testing
    
    # Parameters Required
    vehicle.envelope.ultimate_load                      = 3.5                             # Ultimate load
    vehicle.mass_properties.max_takeoff                 = 79015.8 * Units.kilograms       # Maximum takeoff weight in kilograms
    vehicle.mass_properties.max_zero_fuel               = 79015.8 * 0.9 * Units.kilograms # Maximum zero fuel weight in kilograms
    vehicle.envelope.limit_load                         = 1.5                             # Limit Load
    
    #Build an dsize the turbofan to get sls sthrust
    turbofan = SUAVE.Components.Energy.Networks.Turbofan()
    turbofan.tag = 'turbofan'
    
    # setup
    turbofan.number_of_engines = 2.0
    turbofan.bypass_ratio      = 5.4
    turbofan.engine_length     = 2.71
    turbofan.nacelle_diameter  = 2.05
    
    # working fluid
    turbofan.working_fluid = SUAVE.Attributes.Gases.Air()
    
    
    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    
    # instantiate
    ram = SUAVE.Components.Energy.Converters.Ram()
    ram.tag = 'ram'
    
    # add to the network
    turbofan.append(ram)


    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    
    # add to network
    turbofan.append(inlet_nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 3 - Low Pressure Compressor
    
    # instantiate 
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'low_pressure_compressor'

    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 1.14    
    
    # add to network
    turbofan.append(compressor)

    
    # ------------------------------------------------------------------
    #  Component 4 - High Pressure Compressor
    
    # instantiate
    compressor = SUAVE.Components.Energy.Converters.Compressor()    
    compressor.tag = 'high_pressure_compressor'
    
    # setup
    compressor.polytropic_efficiency = 0.91
    compressor.pressure_ratio        = 13.415    
    
    # add to network
    turbofan.append(compressor)


    # ------------------------------------------------------------------
    #  Component 5 - Low Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='low_pressure_turbine'
    
    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # add to network
    turbofan.append(turbine)
    
      
    # ------------------------------------------------------------------
    #  Component 6 - High Pressure Turbine
    
    # instantiate
    turbine = SUAVE.Components.Energy.Converters.Turbine()   
    turbine.tag='high_pressure_turbine'

    # setup
    turbine.mechanical_efficiency = 0.99
    turbine.polytropic_efficiency = 0.93     
    
    # add to network
    turbofan.append(turbine)
      
    
    # ------------------------------------------------------------------
    #  Component 7 - Combustor
    
    # instantiate    
    combustor = SUAVE.Components.Energy.Converters.Combustor()   
    combustor.tag = 'combustor'
    
    # setup
    combustor.efficiency                = 0.99 
    combustor.alphac                    = 1.0     
    combustor.turbine_inlet_temperature = 1450
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()    
    
    # add to network
    turbofan.append(combustor)

    
    # ------------------------------------------------------------------
    #  Component 8 - Core Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'core_nozzle'
    
    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99    
    
    # add to network
    turbofan.append(nozzle)


    # ------------------------------------------------------------------
    #  Component 9 - Fan Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'fan_nozzle'

    # setup
    nozzle.polytropic_efficiency = 0.95
    nozzle.pressure_ratio        = 0.99    
    
    # add to network
    turbofan.append(nozzle)
    
    
    # ------------------------------------------------------------------
    #  Component 10 - Fan
    
    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    # add to network
    turbofan.append(fan)
    
    
    # ------------------------------------------------------------------
    #Component 10 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag ='compute_thrust'
 
    #total design thrust (includes all the engines)
    thrust.total_design             = 2*24000. * Units.N #Newtons
 
    #design sizing conditions
    altitude      = 35000.0*Units.ft
    mach_number   = 0.78 
    isa_deviation = 0.
    
    # add to network
    turbofan.thrust = thrust

    #size the turbofan
    turbofan_sizing(turbofan,mach_number,altitude)   
    
    # add  gas turbine network gt_engine to the vehicle
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
    wing.sweeps.quarter_chord     = .4363323 * Units.rad       # sweep angle in degrees
    wing.chords.root              = 15.      * Units.meter     # Wing root chord length
    wing.chords.mean_aerodynamic  = 10.      * Units.meters    # Length of the mean aerodynamic chord of the wing
    wing.origin                   = [20,0,0] * Units.meters    # Location of main wing from origin of the vehicle
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
    wing.sweeps.quarter_chord     = 38.     * Units.deg       # Sweep of the horizontal tail
    wing.chords.mean_aerodynamic  = 5.      * Units.meters    # Length of the mean aerodynamic chord of the horizontal tail
    wing.thickness_to_chord       = 0.07                      # Thickness-to-chord ratio of the horizontal tail
    wing.areas.exposed            = 199.7792                  # Exposed area of the horizontal tail
    wing.areas.wetted             = 249.724                   # Wetted area of the horizontal tail
    wing.origin                   = [45,0,0]                  # Location of horizontal tail from origin of the vehicle
    wing.aerodynamic_center       = [3,0,0]                   # Location of aerodynamic center from origin of the horizontal tail
    vehicle.append_component(wing)    
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag = 'vertical_stabilizer'    
    wing.areas.reference      = 60.     * Units.meters**2 # Area of the vertical tail
    wing.spans.projected      = 15.     * Units.meters    # Span of the vertical tail
    wing.thickness_to_chord   = 0.07                      # Thickness-to-chord ratio of the vertical tail
    wing.sweeps.quarter_chord = 40.     * Units.deg       # Sweep of the vertical tail
    wing.t_tail               = "false"                   # Set to "yes" for a T-tail
    vehicle.append_component(wing)   
    
    weight = Tube_Wing.empty(vehicle)
    
    
    actual = Data()
    actual.payload = 17349.9081525
    actual.pax = 15036.5870655
    actual.bag = 2313.321087
    actual.fuel = -13680.6265874
    actual.empty = 75346.5184349
    actual.wing = 27694.192985
    actual.fuselage = 11423.9380852
    actual.propulsion = 6855.68572746 
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
    error.wt_furnish = (actual.wt_furnish - weight.systems_breakdown.furnish)/actual.wt_furnish
    
    error.horizontal_tail = (actual.horizontal_tail - weight.horizontal_tail)/actual.horizontal_tail
    error.vertical_tail = (actual.vertical_tail - weight.vertical_tail)/actual.vertical_tail
    error.rudder = (actual.rudder - weight.rudder)/actual.rudder
    
    print 'Results (kg)'
    print weight
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<0.001)    
   
    
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()    
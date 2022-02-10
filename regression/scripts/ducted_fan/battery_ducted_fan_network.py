# battery_ducted_fan_network.py
# 
# Created:  Apr 2019, C. McMillan
#        

""" create and evaluate a battery ducted fan network
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

import numpy as np

from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.ducted_fan_sizing import ducted_fan_sizing
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------
def main():
    
    # call the network function
    energy_network()
    
    return

def energy_network():
    # ------------------------------------------------------------------
    #   Battery Ducted Fan Network
    # ------------------------------------------------------------------    
    
    #instantiate the ducted fan network
    ducted_fan = SUAVE.Components.Energy.Networks.Ducted_Fan()
    ducted_fan.tag = 'ducted fan'
    
    # setup
    ducted_fan.number_of_engines = 2
    ducted_fan.engine_length     = 1.038 * Units.meter
    ducted_fan.nacelle_diameter  = 1.064 * Units.meter
    ducted_fan.origin            = [[8.95, 1.48, 0.50],[8.95, -1.48, 0.50]] # meters

    #compute engine areas
    ducted_fan.areas.wetted      = 1.1*np.pi*ducted_fan.nacelle_diameter*ducted_fan.engine_length
    
    # working fluid
    ducted_fan.working_fluid = SUAVE.Attributes.Gases.Air()

    # ------------------------------------------------------------------
    #   Component 1 - Ram
    
    # to convert freestream static to stagnation quantities
    # instantiate
    ram         = SUAVE.Components.Energy.Converters.Ram()
    ram.tag     = 'ram'
    
    # add to the network
    ducted_fan.append(ram)

    # ------------------------------------------------------------------
    #  Component 2 - Inlet Nozzle
    
    # instantiate
    inlet_nozzle        = SUAVE.Components.Energy.Converters.Compression_Nozzle()
    inlet_nozzle.tag    = 'inlet_nozzle'
    
    # setup
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    
    # add to network
    ducted_fan.append(inlet_nozzle)

    # ------------------------------------------------------------------
    #  Component 3 - Fan
    
    # instantiate
    fan = SUAVE.Components.Energy.Converters.Fan()   
    fan.tag                   = 'fan'

    # setup
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7    
    
    # add to network
    ducted_fan.append(fan)

    # ------------------------------------------------------------------
    #  Component 4 - Fan Nozzle
    
    # instantiate
    nozzle = SUAVE.Components.Energy.Converters.Expansion_Nozzle()   
    nozzle.tag = 'fan_nozzle'

    # setup
    nozzle.polytropic_efficiency    = 0.95
    nozzle.pressure_ratio           = 0.99  

    # add to network
    ducted_fan.append(nozzle)

    # ------------------------------------------------------------------
    #Component 5 : thrust (to compute the thrust)
    thrust = SUAVE.Components.Energy.Processes.Thrust()       
    thrust.tag                 ='thrust'
 
    #total design thrust (includes all the engines)
    thrust.total_design        = 2*700. * Units.lbf 
    
    # add to network
    ducted_fan.thrust          = thrust   
    
    #design sizing conditions
    altitude      = 0.0 * Units.km
    mach_number   = 0.01
    isa_deviation = 0.
    
    #size the ducted fan
    ducted_fan_sizing(ducted_fan,mach_number,altitude)
    
    battery_ducted_fan                      = SUAVE.Components.Energy.Networks.Battery_Ducted_Fan()
    battery_ducted_fan.tag                  = 'battery_ducted_fan'
    battery_ducted_fan.nacelle_diameter     = ducted_fan.nacelle_diameter
    battery_ducted_fan.areas                = Data()
    battery_ducted_fan.areas.wetted         = ducted_fan.areas.wetted
    battery_ducted_fan.engine_length        = ducted_fan.engine_length
    battery_ducted_fan.origin               = ducted_fan.origin
    battery_ducted_fan.voltage              = 400.

    # add  gas turbine network turbofan to the network 
    battery_ducted_fan.propulsor            = ducted_fan

    # Create ESC and add to the network
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency                  = 0.97
    battery_ducted_fan.esc          = esc

    # Create payload and add to the network
    payload = SUAVE.Components.Energy.Peripherals.Avionics()
    payload.power_draw              = 0.
    payload.mass_properties.mass    = 0. * Units.kg 
    battery_ducted_fan.payload      = payload

    # Create avionics and add to the network
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw             = 200. * Units.watts 
    battery_ducted_fan.avionics     = avionics

    # Create the battery and add to the network
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.specific_energy             = 300. * Units.Wh/Units.kg
    bat.resistance                  = 0.006
    bat.max_voltage                 = 400.
    bat.mass_properties.mass        = 1000. * Units.kg 
    initialize_from_mass(bat, bat.mass_properties.mass)
    battery_ducted_fan.battery      = bat
    
    # ------------------------------------------------------------------
    #   Evaluation Conditions
    # ------------------------------------------------------------------    
    
    # Conditions        
    ones_1col                        = np.array([[1.0],[1.0]])       
    alt                              = 5.0 * Units.km
    
    # Setup the conditions to run the network
    state               = Data()
    state.numerics      = SUAVE.Analyses.Mission.Segments.Conditions.Numerics()
    state.conditions    = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    conditions          = state.conditions
    numerics            = state.numerics
    
    planet              = SUAVE.Attributes.Planets.Earth()   
    working_fluid       = SUAVE.Attributes.Gases.Air()    
    
     # Calculate atmospheric properties
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions =  atmosphere.compute_values(alt)
    rho = atmosphere_conditions.density[0,:]
    a   = atmosphere_conditions.speed_of_sound[0,:]
    mu  = atmosphere_conditions.dynamic_viscosity[0,:]
    T   = atmosphere_conditions.temperature[0,:]
    mach = 0.7

    # aerodynamic conditions
    conditions.expand_rows(2)
    conditions.freestream.mach_number                  = mach * ones_1col
    conditions.freestream.pressure                     = atmosphere_conditions.pressure * ones_1col
    conditions.freestream.temperature                  = T * ones_1col
    conditions.freestream.density                      = rho * ones_1col
    conditions.freestream.dynamic_viscosity            = mu * ones_1col
    conditions.freestream.speed_of_sound               = a * ones_1col
    conditions.freestream.velocity                     = conditions.freestream.mach_number * conditions.freestream.speed_of_sound
    conditions.freestream.altitude                     = alt * ones_1col
    conditions.freestream.gravity                      = planet.compute_gravity(alt) * ones_1col
    conditions.propulsion.battery_energy               = bat.max_energy * ones_1col
    conditions.frames.inertial.time                    = np.array([[0.0],[1.0]])
    conditions.freestream.isentropic_expansion_factor  = ones_1col*working_fluid.compute_gamma(atmosphere_conditions.temperature,atmosphere_conditions.pressure)                                                                                             
    conditions.freestream.Cp                           = ones_1col*working_fluid.compute_cp(atmosphere_conditions.temperature,atmosphere_conditions.pressure)
    conditions.freestream.R                            = ones_1col*working_fluid.gas_specific_constant    
    conditions.q                                       = 0.5 * conditions.freestream.density * conditions.freestream.velocity**2
    # numerics conditions
    numerics.time.integrate                            = np.array([[0, 0],[0, 1]])
    numerics.time.differentiate                        = np.array([[0, 0],[0, 1]])
    numerics.time.control_points                       = np.array([[0], [1]])
    
    # propulsion conditions
    conditions.propulsion.throttle                     = np.array([[1.0],[1.0]])
    conditions.propulsion.battery_energy               = bat.max_energy*np.ones_like(ones_1col)
    
    print("Design thrust ", battery_ducted_fan.propulsor.design_thrust 
          * battery_ducted_fan.number_of_engines)
    print("Sealevel static thrust ", battery_ducted_fan.propulsor.sealevel_static_thrust
          * battery_ducted_fan.number_of_engines)
    
    
    results_off_design  = battery_ducted_fan(state)
    F                   = results_off_design.thrust_force_vector
    mdot                = results_off_design.vehicle_mass_rate
    power               = results_off_design.power
    

    
    #Specify the expected values
    expected        = Data()
    expected.thrust = 3053.50858719
    expected.power  = 849362.83002354
    expected.mdot   = 0.0
    
    #error data function
    error              =  Data()
    error.thrust_error = (F[0][0] -  expected.thrust)/expected.thrust
    error.power_error  = (power[0] - expected.power)/expected.power
    error.mdot_error   = (mdot[0][0] - expected.mdot) # Can't divide by zero
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)    
    
    return
    
if __name__ == '__main__':
    
    main()
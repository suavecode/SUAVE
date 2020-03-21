# test_solar_network.py
# 
# Created:  Aug 2014, Emilio Botero, 
#           Mar 2020, M. Clarke

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

from SUAVE.Core import (
Data, Container,
)

import numpy as np
import copy, time

from SUAVE.Components.Energy.Networks.Solar import Solar
from SUAVE.Methods.Propulsion import propeller_design
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power, initialize_from_mass

def main():

    # ------------------------------------------------------------------
    #   Propulsor
    # ------------------------------------------------------------------
    
    # build network
    net = Solar()
    net.number_of_engines = 1.
    net.nacelle_dia       = 0.2
    
    # Component 1 the Sun?
    sun = SUAVE.Components.Energy.Processes.Solar_Radiation()
    net.solar_flux = sun
    
    # Component 2 the solar panels
    panel = SUAVE.Components.Energy.Converters.Solar_Panel()
    panel.area                 = 100 * Units.m
    panel.efficiency           = 0.18
    panel.mass_properties.mass = panel.area*.600
    net.solar_panel            = panel
    
    # Component 3 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc       = esc
    
    # Component 5 the Propeller
    
    # Propeller design specs
    design_altitude = 0.0 * Units.km
    Velocity        = 10.0  # freestream m/s
    RPM             = 5887
    Blades          = 2.0
    Radius          = .4064
    Hub_Radius      = 0.05
    Design_Cl       = 0.7
    Power           = 7500.  #Specify either thrust or power to design for
    
    # Design the Propeller
    prop                     = SUAVE.Components.Energy.Converters.Propeller()
    prop.number_blades       = Blades 
    prop.freestream_velocity = Velocity
    prop.angular_velocity    = RPM*(2.*np.pi/60.0)
    prop.tip_radius          = Radius
    prop.hub_radius          = Hub_Radius
    prop.design_Cl           = Design_Cl 
    prop.design_altitude     = design_altitude
    prop.design_power        = Power
    prop                     = propeller_design(prop)
    
    # Create and attach this propeller
    net.propeller        = prop
    
    # Component 4 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor()
    motor.resistance           = 0.01
    motor.no_load_current      = 8.0
    motor.speed_constant       = 140.*(2.*np.pi/60.) # RPM/volt converted to rad/s     
    motor.propeller_radius     = prop.tip_radius
    motor.gear_ratio           = 1.
    motor.gearbox_efficiency   = 1.
    motor.expected_current     = 260.
    motor.mass_properties.mass = 2.0
    net.motor                  = motor   
    
    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 0. #Watts 
    payload.mass_properties.mass = 0. * Units.kg
    net.payload                  = payload
    
    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 0. #Watts  
    net.avionics        = avionics      
    
    # Component 8 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    batterymass = 50.  #kg
    bat.type = 'Li-Ion'
    bat.resistance = 0.0
    bat.energy_density = 250.
    initialize_from_mass(bat,batterymass)
    bat.current_energy = bat.max_energy
    net.battery = bat
    
    #Component 9 the system logic controller and MPPT
    logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.system_voltage  = 50.0
    logic.MPPT_efficiency = 0.95
    net.solar_logic       = logic
    
    # Setup the conditions to run the network
    state            = Data()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    state.numerics   = SUAVE.Analyses.Mission.Segments.Conditions.Numerics()
    
    conditions = state.conditions
    numerics   = state.numerics
    
    # Calculate atmospheric properties
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere_conditions =  atmosphere.compute_values(prop.design_altitude)
    
    rho = atmosphere_conditions.density[0,:]
    a   = atmosphere_conditions.speed_of_sound[0,:]
    mu  = atmosphere_conditions.dynamic_viscosity[0,:]
    T   = atmosphere_conditions.temperature[0,:]

    conditions.propulsion.throttle            = np.array([[1.0],[1.0]])
    conditions.freestream.velocity            = np.array([[1.0],[1.0]])
    conditions.freestream.density             = np.array([rho,rho])
    conditions.freestream.dynamic_viscosity   = np.array([mu, mu])
    conditions.freestream.speed_of_sound      = np.array([a, a])
    conditions.freestream.altitude            = np.array([[design_altitude], [design_altitude]])
    conditions.propulsion.battery_energy      = bat.max_energy*np.ones_like(conditions.freestream.altitude)
    conditions.frames.body.inertial_rotations = np.zeros([2,3])
    conditions.frames.inertial.time           = np.array([[0.0],[1.0]])
    numerics.time.integrate                   = np.array([[0, 0],[0, 1]])
    numerics.time.differentiate               = np.array([[0, 0],[0, 1]])
    conditions.frames.planet.start_time       = time.strptime("Sat, Jun 21 06:00:00  2014", "%a, %b %d %H:%M:%S %Y",) 
    conditions.frames.planet.latitude         = np.array([[0.0],[0.0]])
    conditions.frames.planet.longitude        = np.array([[0.0],[0.0]])
    conditions.freestream.temperature         = np.array([T, T])
    conditions.frames.body.transform_to_inertial = np.array([[[ 1.,  0.,  0.],
                                                              [ 0.,  1.,  0.],
                                                              [ 0.,  0.,  1.]],
                                                             [[ 1.,  0.,  0.],
                                                              [ 0.,  1.,  0.],
                                                              [ 0.,  0.,  1.]]])
    conditions.propulsion.propeller_power_coefficient = np.array([[1.], [1.]]) * prop.power_coefficient
    
    # Run the network and print the results
    results = net(state)
    F       = results.thrust_force_vector
    
    # Truth results
    truth_F   = [[360.43835053, 360.43835053]]
    truth_i   = [[ 249.31622624], [ 249.31622624]]
    truth_rpm = [[6668.4094191],[6668.4094191]]
    truth_bat = [[ 36000000.   ], [ 35987534.18868808]]
    
    error = Data()
    error.Thrust = np.max(np.abs(F[:,0]-truth_F))
    error.RPM = np.max(np.abs(conditions.propulsion.rpm-truth_rpm))
    error.Current  = np.max(np.abs(conditions.propulsion.current-truth_i))
    error.Battery = np.max(np.abs(bat.current_energy-truth_bat))
    
    print(error)
    
    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)
        
    
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    
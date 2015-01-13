# test_solar_network.py
# 
# Created:  Emilio Botero, Aug 2014

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

import numpy as np
import copy, time

from SUAVE.Components.Energy.Networks.Solar_Network import Solar_Network
from SUAVE.Methods.Propulsion import propeller_design

def main():

    # ------------------------------------------------------------------
    #   Propulsor
    # ------------------------------------------------------------------
    
    # build network
    net = Solar_Network()
    net.number_motors = 1.
    net.nacelle_dia   = 0.2
    
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
    
    #Propeller design specs
    design_altitude = 0.0 * Units.km
    Velocity        = 10.0  # freestream m/s
    RPM             = 5887
    Blades          = 2.0
    Radius          = .4064
    Hub_Radius      = 0.05
    Design_Cl       = 0.7
    Thrust          = 0.0 #Specify either thrust or power to design for
    Power           = 7500.  #Specify either thrust or power to design for
    
    # Design the Propeller
    prop_attributes = Data()
    prop_attributes.number_blades       = Blades 
    prop_attributes.freestream_velocity = Velocity
    prop_attributes.angular_velocity    = RPM*(2.*np.pi/60.0)
    prop_attributes.tip_radius          = Radius
    prop_attributes.hub_radius          = Hub_Radius
    prop_attributes.design_Cl           = Design_Cl 
    prop_attributes.design_altitude     = design_altitude
    prop_attributes.design_thrust       = Thrust
    prop_attributes.design_power        = Power
    prop_attributes                     = propeller_design(prop_attributes)
    
    # Create and attach this propeller
    prop                 = SUAVE.Components.Energy.Converters.Propeller()
    prop.prop_attributes = prop_attributes
    net.propeller        = prop
    
    # Component 4 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor()
    motor.resistance           = 0.01
    motor.no_load_current      = 8.0
    motor.speed_constant       = 140.*(2.*np.pi/60.) # RPM/volt converted to rad/s     
    motor.propeller_radius     = prop.prop_attributes.tip_radius
    motor.propeller_Cp         = prop.prop_attributes.Cp
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
    bat = SUAVE.Components.Energy.Storages.Battery()
    bat.mass_properties.mass = 50.  #kg
    bat.type = 'Li-Ion'
    bat.resistance = 0.0
    net.battery = bat
    
    #Component 9 the system logic controller and MPPT
    logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.system_voltage  = 50.0
    logic.MPPT_efficiency = 0.95
    net.solar_logic       = logic
    
    # Setup the conditions to run the network
    conditions                 = Data()
    conditions.propulsion      = Data()
    conditions.freestream      = Data()
    conditions.frames          = Data()
    conditions.frames.body     = Data()
    conditions.frames.inertial = Data()
    conditions.frames.planet   = Data()
    numerics                   = Data()
    
    # Calculate atmospheric properties
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    p, T, rho, a, mu = atmosphere.compute_values(design_altitude)
    
    conditions.propulsion.throttle            = np.array([[1.0],[1.0]])
    conditions.freestream.velocity            = np.array([[1.0],[1.0]])
    conditions.freestream.density             = np.array([rho,rho])
    conditions.freestream.viscosity           = np.array([mu, mu])
    conditions.freestream.speed_of_sound      = np.array([a, a])
    conditions.freestream.altitude            = np.array([[design_altitude], [design_altitude]])
    conditions.propulsion.battery_energy      = bat.max_energy()*np.ones_like(conditions.freestream.altitude)
    conditions.frames.body.inertial_rotations = np.zeros([2,3])
    conditions.frames.inertial.time           = np.array([[0.0],[1.0]])
    numerics.integrate_time                   = np.array([[0, 0],[0, 1]])
    conditions.frames.planet.start_time       = time.strptime("Sat, Jun 21 06:00:00  2014", "%a, %b %d %H:%M:%S %Y",) 
    conditions.frames.planet.latitude         = np.array([[0.0],[0.0]])
    conditions.frames.planet.longitude        = np.array([[0.0],[0.0]])
    conditions.freestream.temperature         = np.array([T, T])
    
    # Run the network and print the results
    F, mdot, P = net(conditions,numerics)
    
    # Truth results
    truth_F   = [[538.00449442], [538.00449442]]
    truth_P   = [[14272.1902522],[14272.1902522]]
    truth_i   = [[ 249.31622624],[ 249.31622624]]
    truth_rpm = [[ 6668.4094191],[ 6668.4094191]]
    truth_bat = [[45000000.] , [44987534.18868808]]
    
    error = Data()
    error.Thrust = np.max(np.abs(F-truth_F))
    error.Propeller_Power   = np.max(np.abs(P-truth_P))
    error.RPM = np.max(np.abs(conditions.propulsion.rpm-truth_rpm))
    error.Current  = np.max(np.abs(conditions.propulsion.current-truth_i))
    error.Battery = np.max(np.abs(bat.CurrentEnergy-truth_bat))
    
    print  error
    
    for k,v in error.items():
        assert(np.abs(v)<0.001)
        
    
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    
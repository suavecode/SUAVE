# solar_low_fidelity_network.py
# 
# Created: Apr 2018, W. Maier 
#          Mar 2020, M. Clarke
#          Apr 2020, E. Botero

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data


import numpy as np
import time

from SUAVE.Components.Energy.Networks.Solar_Low_Fidelity import Solar_Low_Fidelity
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_mass
from SUAVE.Methods.Propulsion.electric_motor_sizing import size_from_kv

def main():
   
    #------------------------------------------------------------------
    # Network
    #------------------------------------------------------------------

    # build network
    net = Solar_Low_Fidelity()
    net.number_of_engines = 1.
    net.nacelle_diameter  = 0.05
    net.areas             = Data()
    net.areas.wetted      = 0.01*(2*np.pi*0.01/2)
    net.engine_length     = 0.01

    # Component 1 the Sun
    sun = SUAVE.Components.Energy.Processes.Solar_Radiation()
    net.solar_flux = sun

    # Component 2 the solar panels
    panel = SUAVE.Components.Energy.Converters.Solar_Panel()
    panel.ratio                = 0.9
    panel.area                 = 1.0 * panel.ratio 
    panel.efficiency           = 0.25
    panel.mass_properties.mass = panel.area*(0.60 * Units.kg)
    net.solar_panel            = panel

    # Component 3 the ESC
    esc = SUAVE.Components.Energy.Distributors.Electronic_Speed_Controller()
    esc.efficiency = 0.95 # Gundlach for brushless motors
    net.esc        = esc

    # Component 5 the Propeller
    prop = SUAVE.Components.Energy.Converters.Propeller_Lo_Fid()
    prop.propulsive_efficiency = 0.825
    net.propeller        = prop
    
    # Component 4 the Motor
    motor = SUAVE.Components.Energy.Converters.Motor_Lo_Fid() 
    motor.speed_constant       = 800. * Units['rpm/volt'] # RPM/volt is standard
    motor                      = size_from_kv(motor)    
    motor.gear_ratio           = 1. # Gear ratio, no gearbox
    motor.gearbox_efficiency   = 1. # Gear box efficiency, no gearbox
    motor.motor_efficiency     = 0.825;
    net.motor                  = motor    

    # Component 6 the Payload
    payload = SUAVE.Components.Energy.Peripherals.Payload()
    payload.power_draw           = 0. #Watts 
    payload.mass_properties.mass = 0.0 * Units.kg
    net.payload                  = payload

    # Component 7 the Avionics
    avionics = SUAVE.Components.Energy.Peripherals.Avionics()
    avionics.power_draw = 10. #Watts  
    net.avionics        = avionics      

    # Component 8 the Battery
    bat = SUAVE.Components.Energy.Storages.Batteries.Constant_Mass.Lithium_Ion()
    bat.mass_properties.mass = 5.0  * Units.kg
    bat.specific_energy      = 250. *Units.Wh/Units.kg
    bat.resistance           = 0.003
    bat.iters                = 0
    initialize_from_mass(bat)   
    net.battery              = bat

    #Component 9 the system logic controller and MPPT
    logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
    logic.system_voltage  = 18.5
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
    atmosphere_conditions =  atmosphere.compute_values(1000.*Units.ft)
    
    rho = atmosphere_conditions.density[0,:]
    a   = atmosphere_conditions.speed_of_sound[0,:]
    mu  = atmosphere_conditions.dynamic_viscosity[0,:]
    T   = atmosphere_conditions.temperature[0,:]

    conditions.propulsion.throttle            = np.array([[1.0],[1.0]])
    conditions.freestream.velocity            = np.array([[1.0],[1.0]])
    conditions.freestream.density             = np.array([rho,rho])
    conditions.freestream.dynamic_viscosity   = np.array([mu, mu])
    conditions.freestream.speed_of_sound      = np.array([a, a])
    conditions.freestream.altitude            = np.array([[1000.0],[1000.0]])
    conditions.propulsion.battery_energy      = bat.max_energy*np.ones_like(conditions.freestream.altitude)
    conditions.frames.body.inertial_rotations = np.zeros([2,3])
    conditions.frames.inertial.time           = np.array([[0.0],[1.0]])
    numerics.time.integrate                   = np.array([[0, 0],[0, 1]])
    numerics.time.differentiate               = np.array([[0, 0],[0, 1]])
    numerics.time.control_points              = np.array([[0, 0],[0, 1]]) 
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
    # Run the network and print the results
    results = net(state)
    F       = results.thrust_force_vector
    
    # Truth results
    truth_F   = [[68.78277813       ], [ 68.78277813     ]]
    truth_i   = [[ 5.75011436       ], [ 5.75011436      ]]
    truth_rpm = [[ 14390.30435183   ], [ 14390.30435183  ]]
    truth_bat = [[3169014.08450704  ], [3168897.35916947 ]]
    
    error = Data()
    error.Thrust = np.max(np.abs(F[:,0]-truth_F))
    error.RPM = np.max(np.abs(conditions.propulsion.propeller_rpm-truth_rpm))
    error.Current  = np.max(np.abs(conditions.propulsion.battery_current-truth_i))
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
    
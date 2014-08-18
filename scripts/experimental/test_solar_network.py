# test_solar_network.py
# 
# Created:  Emilio Botero, Aug 2014

#----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import sys
sys.path.append('../trunk')


import SUAVE
from SUAVE.Attributes import Units

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

import numpy as np
import pylab as plt
import matplotlib
import copy, time

from SUAVE.Components.Energy.Networks.Solar_Network import Solar_Network
from SUAVE.Components.Energy.Converters.Propeller_Design import Propeller_Design


# ------------------------------------------------------------------
#   Propulsor
# ------------------------------------------------------------------

#Propeller design specs
design_altitude = 0.0 * Units.km
Velocity        = 30.0  # freestream m/s
RPM             = 5887.
Blades          = 2.0
Radius          = .4064
Hub_Radius      = 0.02
Thrust          = 0.0    #Specify either thrust or power to design for
Power           = 12042.0 #Specify either thrust or power to design for
atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
p, T, rho, a, mu = atmosphere.compute_values(design_altitude)

#Design the Propeller
Prop_attributes = Data()
Prop_attributes.nu     = mu/rho
Prop_attributes.B      = Blades 
Prop_attributes.V      = Velocity
Prop_attributes.omega  = RPM*(2.*np.pi/60.0)
Prop_attributes.R      = Radius
Prop_attributes.Rh     = Hub_Radius
Prop_attributes.Des_CL = 0.7
Prop_attributes.rho    = rho
Prop_attributes.Tc     = 2.*Thrust/(rho*(Velocity**2.)*np.pi*(Radius**2.))
Prop_attributes.Pc     = 2.*Power/(rho*(Velocity**3.)*np.pi*(Radius**2.))
Prop_attributes        = Propeller_Design(Prop_attributes)

# build network
net = Solar_Network()
net.num_motors = 1.
net.nacelle_dia = 0.2

# Component 1 the Sun?
sun = SUAVE.Components.Energy.Properties.solar()
net.solar_flux = sun

# Component 2 the solar panels
panel = SUAVE.Components.Energy.Converters.Solar_Panel()
panel.A = 10.
panel.eff = 0.22
net.solar_panel = panel

# Component 3 the ESC
esc = SUAVE.Components.Energy.Distributors.ESC()
esc.eff = 0.95 # Gundlach for brushless motors
net.esc = esc

# Component 5 the Propeller
prop = SUAVE.Components.Energy.Converters.Propeller()
prop.Prop_attributes = Prop_attributes
net.propeller = prop

# Component 4 the Motor
motor = SUAVE.Components.Energy.Converters.Motor()
motor.Res = 0.006
motor.io = 10
motor.kv = 145.*(2.*np.pi/60.) # RPM/volt converted to rad/s     
motor.propradius = prop.Prop_attributes.R
motor.propCp = prop.Prop_attributes.Cp
motor.G   = 1. # Gear ratio
motor.etaG = 1. # Gear box efficiency
motor.exp_i = 260. # Expected current
net.motor = motor    

# Component 6 the Payload
payload = SUAVE.Components.Energy.Sinks.Payload()
payload.draw = 0. #Watts 
payload.Mass_Props.mass = 0.0 * Units.kg
net.payload = payload

# Component 7 the Avionics
avionics = SUAVE.Components.Energy.Sinks.Avionics()
avionics.draw = 0. #Watts  
net.avionics = avionics

# Component 8 the Battery
bat = SUAVE.Components.Energy.Storages.Battery()
bat.Mass_Props.mass = 50.  #kg
bat.type = 'Li-Ion'
bat.R0 = 0.07446
net.battery = bat

#Component 9 the system logic controller and MPPT
logic = SUAVE.Components.Energy.Distributors.Solar_Logic()
logic.systemvoltage = 50.0
logic.MPPTeff = 0.95
net.solar_logic = logic

#Setup the conditions to run the network
conditions                 = Data()
conditions.propulsion      = Data()
conditions.freestream      = Data()
conditions.frames          = Data()
conditions.frames.body     = Data()
conditions.frames.inertial = Data()
numerics                   = Data()

conditions.propulsion.throttle            = np.transpose(np.array([[1.0, 1.0]]))
conditions.freestream.velocity            = np.transpose(np.array([[1.0, 1.0]]))
conditions.freestream.density             = np.array([rho, rho])
conditions.freestream.viscosity           = np.transpose(np.array([[mu, mu]]))
conditions.freestream.speed_of_sound      = np.transpose(np.array([[a, a]]))
conditions.freestream.altitude            = np.transpose(np.array([[design_altitude, design_altitude]]))
conditions.propulsion.battery_energy      = bat.max_energy()*np.ones_like(conditions.freestream.altitude)
conditions.frames.body.inertial_rotations = np.ones([2,3]) * 0.0
conditions.frames.inertial.time           = np.transpose(np.array([[0.0, 1.0]]))
numerics.integrate_time                   = np.array([[0, 0], [0, 1]])

#Run the network and print the results
F, mdot, P = net(conditions,numerics)

print F/9.81
print P

print conditions.propulsion.current
print conditions.propulsion.rpm
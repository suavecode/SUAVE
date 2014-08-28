#Solar_Network.py
# 
# Created:  Emilio Botero, Jun 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Attributes import Units

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Solar_Network(Data):
    def __defaults__(self):
        self.solar_flux  = None
        self.solar_panel = None
        self.motor       = None
        self.propeller   = None
        self.esc         = None
        self.avionics    = None
        self.payload     = None
        self.solar_logic = None
        self.battery     = None
        self.nacelle_dia = 0.0
        self.tag         = 'Network'
    
    # manage process with a driver function
    def evaluate(self,conditions,numerics):
    
        # unpack
        solar_flux  = self.solar_flux
        solar_panel = self.solar_panel
        motor       = self.motor
        propeller   = self.propeller
        esc         = self.esc
        avionics    = self.avionics
        payload     = self.payload
        solar_logic = self.solar_logic
        battery     = self.battery
        

        #================================
        # Lat/Lon
        #================================
        
        # Time of the mission start
        conditions.frames.planet.timedate  = time.strptime("Sat, Jun 21 12:30:00  2014", "%a, %b %d %H:%M:%S %Y",)  
        
        # Unpack some conditions
        V          = conditions.freestream.velocity[:,0]
        altitude   = conditions.freestream.altitude[:,0]
        phi        = conditions.frames.body.inertial_rotations[:,0]
        theta      = conditions.frames.body.inertial_rotations[:,1]
        psi        = conditions.frames.body.inertial_rotations[:,2]
        I          = numerics.integrate_time
        alpha      = conditions.aerodynamics.angle_of_attack[:,0]
        earthstuff = SUAVE.Attributes.Planets.Earth()
        Re         = earthstuff.mean_radius   
        
        gamma     = theta - alpha
        R         = altitude + Re
        lamdadot  = (V/R)*np.cos(gamma)*np.cos(psi)
        lamda     = np.dot(I,lamdadot) / Units.deg # Latitude
        mudot     = (V/R)*np.cos(gamma)*np.sin(psi)/np.cos(lamda)
        mu        = np.dot(I,mudot) / Units.deg # Longitude

        shape     = np.shape(conditions.freestream.velocity)
        mu        = np.reshape(mu,shape) 
        lamda     = np.reshape(lamda,shape)

        lat = conditions.frames.planet.latitude[0,0]
        lon = conditions.frames.planet.longitude[0,0]
        
        conditions.frames.planet.latitude  = lat + lamda
        conditions.frames.planet.longitude = lon + mu    
        
        #===================================
        #
        #===================================
       
        # Set battery energy
        battery.CurrentEnergy = conditions.propulsion.battery_energy
        
        # step 1
        solar_flux.solar_flux(conditions)
        # link
        solar_panel.inputs.flux = solar_flux.outputs.flux
        # step 2
        solar_panel.power()
        # link
        solar_logic.inputs.powerin = solar_panel.outputs.power
        # step 3
        solar_logic.voltage()
        # link
        esc.inputs.voltagein =  solar_logic.outputs.systemvoltage
        # Step 4
        esc.voltageout(conditions)
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        # step 5
        motor.omega(conditions)
        # link
        propeller.inputs.omega =  motor.outputs.omega
        # step 6
        F, Q, P, Cplast = propeller.spin(conditions)
       
        # iterate the Cp here
        diff = abs(Cplast-motor.propCp)
        tol = 1e-6
        
        while (np.any(diff>tol)):
            motor.propCp = Cplast #Change the Cp
            motor.omega(conditions) #Rerun the motor
            propeller.inputs.omega =  motor.outputs.omega #Relink the motor
            F, Q, P, Cplast = propeller.spin(conditions) #Run the motor again
            diff = abs(Cplast-motor.propCp) #Check to see if it converged
            
        # Run the avionics
        avionics.power()
        # link
        solar_logic.inputs.pavionics =  avionics.outputs.power
        # Run the payload
        payload.power()
        # link
        solar_logic.inputs.ppayload = payload.outputs.power
        # Run the motor for current
        motor.current(conditions)
        # link
        esc.inputs.currentout =  motor.outputs.current
        # Run the esc
        esc.currentin()
        # link
        solar_logic.inputs.currentesc = esc.outputs.currentin*self.num_motors
        #
        solar_logic.logic(conditions,numerics)
        # link
        battery.inputs.batlogic = solar_logic.outputs.batlogic
        battery.energy_calc(numerics)
        
        #Pack the conditions for outputs
        rpm                                  = motor.outputs.omega*60./(2.*np.pi)
        current                              = solar_logic.inputs.currentesc
        battery_draw                         = battery.inputs.batlogic.pbat
        battery_energy                       = battery.CurrentEnergy
        
        conditions.propulsion.solar_flux     = solar_flux.outputs.flux  
        conditions.propulsion.rpm            = np.reshape(rpm,np.shape(solar_flux.outputs.flux))
        conditions.propulsion.current        = np.reshape(current,np.shape(solar_flux.outputs.flux))
        conditions.propulsion.battery_draw   = np.reshape(battery_draw,np.shape(solar_flux.outputs.flux))
        conditions.propulsion.battery_energy = np.reshape(battery_energy,np.shape(solar_flux.outputs.flux))
        
        #Create the outputs
        F    = self.num_motors * F
        mdot = np.zeros_like(F)
        P    = self.num_motors * P
        
        return F, mdot, P
            
    __call__ = evaluate
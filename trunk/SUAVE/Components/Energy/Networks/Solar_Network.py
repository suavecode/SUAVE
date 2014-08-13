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
        
        #Time and location
        conditions.frames.planet          = Data()
        conditions.frames.planet.lat      = 37.4300
        conditions.frames.planet.lon      = -122.1700
        conditions.frames.planet.timedate = time.strptime("Sat, Jun 21 08:30:00  2014", "%a, %b %d %H:%M:%S %Y",)  
        
        ##Set battery energy
        battery.CurrentEnergy = battery.max_energy()*np.ones_like(numerics.time)
        #battery.CurrentEnergy
        
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
       
        #iterate the Cp here
        diff = abs(Cplast-motor.propCp)
        tol = 1e-8
        
        while (np.any(diff>tol)):
            motor.propCp = Cplast #Change the Cp
            motor.omega(conditions) #Rerun the motor
            propeller.inputs.omega =  motor.outputs.omega #Relink the motor
            F, Q, P, Cplast = propeller.spin(conditions) #Run the motor again
            diff = abs(Cplast-motor.propCp) #Check to see if it converged
            
        #Run the avionics
        avionics.power()
        # link
        solar_logic.inputs.pavionics =  avionics.outputs.power
        #Run the payload
        payload.power()
        # link
        solar_logic.inputs.ppayload = payload.outputs.power
        #Run the motor for current
        motor.current(conditions)
        # link
        esc.inputs.currentout =  motor.outputs.current
        #Run the esc
        esc.currentin()
        # link, I cheated here
        solar_logic.inputs.currentesc = esc.outputs.currentin*self.num_motors
        #
        solar_logic.logic(conditions,numerics)
        # link
        battery.inputs.batlogic = solar_logic.outputs.batlogic
        battery.energy_calc(numerics)
        
        #Pack the conditions for outputs
        conditions.propulsion.solar_flux     = solar_flux.outputs.flux  
        rpm                                  = motor.outputs.omega*60./(2.*np.pi)
        conditions.propulsion.rpm            = np.reshape(rpm,np.shape(solar_flux.outputs.flux))
        current                              = solar_logic.inputs.currentesc
        conditions.propulsion.current        = np.reshape(current,np.shape(solar_flux.outputs.flux))
        battery_draw                         = battery.inputs.batlogic.pbat
        conditions.propulsion.battery_draw   = np.reshape(battery_draw,np.shape(solar_flux.outputs.flux))
        battery_energy                       = battery.CurrentEnergy
        conditions.propulsion.battery_energy = np.reshape(battery_energy,np.shape(solar_flux.outputs.flux))
        
        mdot = np.zeros_like(F)

        F = self.num_motors * F
        P = self.num_motors * P
        
        return F, mdot, P
            
    __call__ = evaluate
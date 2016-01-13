#Battery_Propeller.py
# 
# Created: Jul 2015, E. Botero
# Modified: Jul 2015, M. Kruger

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
from SUAVE.Core import Units
from SUAVE.Components.Propulsors.Propulsor import Propulsor

from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Battery_Propeller(Propulsor):
    def __defaults__(self): 
        self.motor             = None
        self.propeller         = None
        self.esc               = None
        self.avionics          = None
        self.payload           = None
        self.battery           = None
        self.nacelle_diameter  = None
        self.engine_length     = None
        self.number_of_engines = None
        self.voltage           = None
        self.thrust_angle      = 0.0
        self.tag               = 'network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
    
        # unpack
        conditions  = state.conditions
        numerics    = state.numerics
        motor       = self.motor
        propeller   = self.propeller
        esc         = self.esc
        avionics    = self.avionics
        payload     = self.payload
        battery     = self.battery
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy  
        battery.voltage        = conditions.propulsion.battery_voltage
        
        diff_v = battery.voltage * 1.
        v_last = battery.voltage * 1.
        tol_v  = 1e-4
        
        while (np.any(diff_v>tol_v)):
       
            # Step 1 battery power
            esc.inputs.voltagein = battery.voltage
            # Step 2
            esc.voltageout(conditions)
            # link
            motor.inputs.voltage = esc.outputs.voltageout 
            # step 3
            motor.omega(conditions)
            # link
            propeller.inputs.omega =  motor.outputs.omega
            propeller.thrust_angle = self.thrust_angle
            # step 4
            F, Q, P, Cplast = propeller.spin(conditions)
           
            # iterate the Cp here
            diff = abs(Cplast-motor.propeller_Cp)
            tol = 1e-6
            ii = 0 
            while (np.any(diff>tol)):
                motor.propeller_Cp  = Cplast #Change the Cp
                motor.omega(conditions) #Rerun the motor
                propeller.inputs.omega =  motor.outputs.omega #Relink the motor
                F, Q, P, Cplast = propeller.spin(conditions) #Run the motor again
                diff = abs(Cplast-motor.propeller_Cp) #Check to see if it converged
                ii += 1
                #if ii>100:
                    #break            
            
                
            # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
            eta = conditions.propulsion.throttle[:,0,None]
            P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
            F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
    
            # Run the avionics
            avionics.power()
    
            # Run the payload
            payload.power()
            
            # Run the motor for current
            motor.current(conditions)
            # link
            esc.inputs.currentout =  motor.outputs.current
            
            # Run the esc
            esc.currentin()
    
            # Calculate avionics and payload power
            avionics_payload_power = avionics.outputs.power + payload.outputs.power
    
            # Calculate avionics and payload current
            avionics_payload_current = avionics_payload_power/self.voltage
    
            # link
            battery.inputs.current = esc.outputs.currentin*self.number_of_engines + avionics_payload_current
            battery.inputs.power_in = -(esc.outputs.voltageout*esc.outputs.currentin*self.number_of_engines + avionics_payload_power)
            battery.energy_calc(numerics)
            
            diff_v = np.abs(v_last-battery.voltage)
            v_last = battery.voltage * 1.            
        
        
        
        
        
        #Pack the conditions for outputs
        rpm                                   = motor.outputs.omega*60./(2.*np.pi)
        current                               = esc.outputs.currentin
        battery_draw                          = battery.inputs.power_in 
        battery_energy                        = battery.current_energy
        battery_voltage                       = battery.voltage
         
        conditions.propulsion.rpm             = rpm
        conditions.propulsion.current         = current
        conditions.propulsion.battery_draw    = battery_draw
        conditions.propulsion.battery_energy  = battery_energy
        conditions.propulsion.battery_voltage = battery_voltage
        
        #Create the outputs
        F    = self.number_of_engines * F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]      
        mdot = np.zeros_like(F)

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        
        return results
            
    __call__ = evaluate_thrust

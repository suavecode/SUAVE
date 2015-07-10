#Battery_Propeller.py
# 
# Created: Jul 2015, E. Botero
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
        self.battery           = None
        self.nacelle_diameter  = None
        self.engine_length     = None
        self.number_of_engines = None
        self.tag               = 'network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
    
        # unpack
        conditions  = state.conditions
        numerics    = state.numerics
        motor       = self.motor
        propeller   = self.propeller
        esc         = self.esc
        battery     = self.battery
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy        
       
        # Step 1 battery power
        esc.inputs.voltagein = self.voltage
        # Step 2
        esc.voltageout(conditions)
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        # step 3
        motor.omega(conditions)
        # link
        propeller.inputs.omega =  motor.outputs.omega
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
        
        # Run the motor for current
        motor.current(conditions)
        # link
        esc.inputs.currentout =  motor.outputs.current
        
        # Run the esc
        esc.currentin()
        # link
        battery.inputs.current  = esc.outputs.currentin*self.number_of_engines
        battery.inputs.power_in = -esc.outputs.voltageout*battery.inputs.current
        battery.energy_calc(numerics)
        
        #Pack the conditions for outputs
        rpm                                  = motor.outputs.omega*60./(2.*np.pi)
        current                              = esc.outputs.currentin
        battery_draw                         = battery.inputs.power_in 
        battery_energy                       = battery.current_energy
         
        conditions.propulsion.rpm            = rpm
        conditions.propulsion.current        = current
        conditions.propulsion.battery_draw   = battery_draw
        conditions.propulsion.battery_energy = battery_energy
        
        #Create the outputs
        F    = self.number_of_engines * F * [1,0,0]      
        mdot = np.zeros_like(F)

        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
        
        return results
            
    __call__ = evaluate_thrust
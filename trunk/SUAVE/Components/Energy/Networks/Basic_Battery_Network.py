#Lithium_Air_Network.py
# 
# Created:  Michael Vegh, September 2014
# Modified:  
'''
Simply connects a battery to a ducted fan, with an assumed motor efficiency
'''
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
#import time
from SUAVE.Attributes import Units

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Basic_Battery_Network(Data):
    def __defaults__(self):

        #self.motor       = None
        self.propulsor   = None
        #self.esc         = None
        #self.avionics    = None
        #self.payload     = None
        #self.solar_logic = None
        self.battery     = None
        self.motor_efficiency=.95 #choose 95% efficiency as default energy conversion efficiency
        #self.nacelle_dia = 0.0
        self.tag         = 'Network'
    
    # manage process with a driver function
    def evaluate(self,conditions,numerics):
        
        # unpack

        propulsor   = self.propulsor
        battery     = self.battery
        I=numerics.integrate_time
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy
        
        
        F, mdot, Pe =propulsor(conditions)
       
        #pbat=-F*conditions.freestream.velocity[0,0]/self.motor_efficiency #power required from the battery
        pbat=np.multiply(-F, conditions.freestream.velocity)/self.motor_efficiency
       
        
        e = np.dot(I,pbat)  #integrate energy required from ducted fan/motor
        batlogic      = Data()
        batlogic.pbat = pbat
        batlogic.Ibat = 90.  #use 90 amps as a default for now; will change this for higher fidelity methods
        batlogic.e    = e
        battery.inputs.batlogic=batlogic
        
        tol = 1e-6
        
        
        if battery.type=='Li-Air':
            mdot=battery.energy_calc(numerics)
        else:
            battery.energy_calc(numerics)
        #Pack the conditions for outputs
        
        battery_draw                         = battery.inputs.batlogic.pbat
        battery_energy                       = battery.current_energy
        
        #conditions.propulsion.solar_flux     = solar_flux.outputs.flux  
        
        conditions.propulsion.battery_draw   = battery_draw
        conditions.propulsion.battery_energy = battery_energy
        #number_of_engines
        #Create the outputs
        
        '''
        F    = propulsor.number_of_engines * F
        
        P    = propulsor.number_of_engines * P
        '''
        
        return F, mdot, pbat
            
    __call__ = evaluate
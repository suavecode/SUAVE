#Basic_Battery_Network.py
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
from SUAVE.Core import Units
from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate
from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Basic_Battery(Data):
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
    
        # Set battery energy
  
        F, mdot, Pe =propulsor(conditions)
       
        try:
            initial_energy=conditions.propulsion.battery_energy
          
            if initial_energy[0]==0: #beginning of segment; initialize battery
                battery.current_energy=battery.current_energy[-1]*np.ones_like(initial_energy)
        except AttributeError: #battery energy not initialized, e.g. in takeoff
            battery.current_energy=battery.current_energy[-1]*np.ones_like(F)
       
        pbat=np.multiply(-F, conditions.freestream.velocity)/self.motor_efficiency
        
        battery_logic     = Data()
        battery_logic.power_in = pbat
        battery_logic.current  = 90.  #use 90 amps as a default for now; will change this for higher fidelity methods
      
        battery.inputs    =battery_logic
        battery.inputs.power_in=pbat
        tol = 1e-6
        battery.energy_calc(numerics)
        #allow for mass gaining batteries
       
        try:
            mdot=find_mass_gain_rate(battery,-(pbat-battery.resistive_losses))
        except AttributeError:
            mdot=np.zeros_like(F)
           
       
        
        
        #Pack the conditions for outputs
        battery_draw                         = battery.inputs.power_in
        battery_energy                       = battery.current_energy
      
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
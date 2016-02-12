# Battery_Ducted_Fan.py
#
# Created:  Sep 2014, M. Vegh
# Modified: Jan 2016, T. MacDonald

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
from SUAVE.Core import Data
from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate
from SUAVE.Components.Propulsors.Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Battery_Ducted_Fan(Propulsor):
    def __defaults__(self):
        self.propulsor        = None
        self.battery          = None
        self.motor_efficiency = .95 
        self.tag              = 'Network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        
        # unpack

        propulsor   = self.propulsor
        battery     = self.battery
    
        conditions  = state.conditions
        numerics    = state.numerics
  
        results = propulsor.evaluate_thrust(state)
        Pe      = np.multiply(results.thrust_force_vector[:,0],conditions.freestream.velocity[0])
        
        try:
            initial_energy = conditions.propulsion.battery_energy
            if initial_energy[0][0]==0: #beginning of segment; initialize battery
                battery.current_energy = battery.current_energy[-1]*np.ones_like(initial_energy)
        except AttributeError: #battery energy not initialized, e.g. in takeoff
            battery.current_energy=np.transpose(np.array([battery.current_energy[-1]*np.ones_like(Pe)]))
        
        pbat = -Pe/self.motor_efficiency
        battery_logic          = Data()
        battery_logic.power_in = pbat
        battery_logic.current  = 90.  #use 90 amps as a default for now; will change this for higher fidelity methods
      
        battery.inputs = battery_logic
        tol = 1e-6
        
        battery.energy_calc(numerics)
        #allow for mass gaining batteries
       
        try:
            mdot=find_mass_gain_rate(battery,-(pbat-battery.resistive_losses)) #put in transpose for solver
        except AttributeError:
            mdot=np.zeros_like(results.thrust_force_vector[:,0])
        mdot=np.reshape(mdot, np.shape(conditions.freestream.velocity))
        #Pack the conditions for outputs
        battery_draw                         = battery.inputs.power_in
        battery_energy                       = battery.current_energy
      
        conditions.propulsion.battery_draw   = battery_draw
        conditions.propulsion.battery_energy = battery_energy
        
        results.vehicle_mass_rate   = mdot
        return results
            
    __call__ = evaluate_thrust

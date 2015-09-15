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
from SUAVE.Components.Propulsors.Propulsor import Propulsor
from SUAVE.Core import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Battery_Ducted_Fan_Parallel_Hybrid(Propulsor):
    def __defaults__(self):
        self.propulsor            = None
        self.primary_battery      = None  #main battery (generally high esp)
        self.auxiliary_battery    = None #used to meet power demands beyond primary
        self.motor_efficiency     =.95 #choose 95% efficiency as default energy conversion efficiency
    
        self.tag         = 'Network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        
        # unpack

        propulsor           = self.propulsor
        primary_battery     = self.primary_battery
        auxiliary_battery  = self.auxiliary_battery
        conditions = state.conditions
        numerics   = state.numerics
  
        results=propulsor.evaluate_thrust(state)
        Pe     =results.thrust_force_vector[:,0]*conditions.freestream.velocity
        
        try:
            initial_energy=conditions.propulsion.primary_battery_energy
            initial_energy_auxiliary=conditions.propulsion.auxiliary_battery_energy
            if initial_energy[0][0]==0: #beginning of segment; initialize battery
                primary_battery.current_energy  =primary_battery.current_energy[-1]*np.ones_like(initial_energy)
                auxiliary_battery.current_energy=auxiliary_battery.current_energy[-1]*np.ones_like(initial_energy)
        except AttributeError: #battery energy not initialized, e.g. in takeoff
            primary_battery.current_energy=primary_battery.current_energy[-1]*np.ones_like(F)
            auxiliary_battery.current_energy=auxiliary_battery.current_energy[-1]*np.ones_like(F)
        
        pbat=-Pe/self.motor_efficiency
        pbat_primary=pbat
        pbat_auxiliary=0.
        if pbat>primary_battery.max_power:   #limit power output of primary_battery
            pbat_primary=primary_battery.max_power
            pbat_auxiliary=pbat-pbat_primary
        primary_battery_logic           = Data()
        primary_battery_logic.power_in  = pbat_primary
        primary_battery_logic.current   = 90.  #use 90 amps as a default for now; will change this for higher fidelity methods
        auxiliary_battery_logic         =copy.copy(primary_battery_logic)
        auxiliary_battery_logic.power_in=pbat_auxiliary
        primary_battery.inputs          =battery_logic
        auxiliary_battery.inputs        =auxiliary_battery_logic
        tol = 1e-6
        primary_battery.energy_calc(numerics)
        auxiliary_battery.energy_calc(numerics)
        #allow for mass gaining batteries
       
        try:
            mdot_primary=find_mass_gain_rate(primary_battery,-(pbat_primary-primary_battery.resistive_losses))
        except AttributeError:
            mdot_primary=np.zeros_like(F)   
        try:
            mdot_auxiliary=find_mass_gain_rate(auxiliary_battery,-(pbat_auxiliary-auxiliary_battery.resistive_losses))
        except AttributeError:
            mdot_auxiliary=np.zeros_like(F)
        
        mdot=mdot_primary+mdot_auxiliary
        
        #Pack the conditions for outputs
        primary_battery_draw                 = primary_battery.inputs.power_in
        primary_battery_energy               = primary_battery.current_energy
        auxiliary_battery_draw               = auxiliary_battery.inputs.power_in
        auxiliary_battery_energy             = auxiliary_battery.current_energy
      
      
      
        conditions.propulsion.primary_battery_draw   = primary_battery_draw
        conditions.propulsion.primary_battery_energy = primary_battery_energy
        
        conditions.propulsion.auxiliary_battery_draw   = auxiliary_battery_draw
        conditions.propulsion.auxiliary_battery_energy = auxiliary_battery_energy
        
     
 
        results.vehicle_mass_rate   = mdot
        return results
            
    __call__ = evaluate_thrust
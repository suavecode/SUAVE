## @ingroup Components-Energy-Networks
# Dual_Battery_Ducted_Fan.py
#
# Created:  Sep 2015, M. Vegh
# Modified: Feb 2016, M. Vegh
#           Jan 2016, T. MacDonald
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Core import Data
from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate
from SUAVE.Components.Propulsors.Propulsor import Propulsor


# package imports
import numpy as np
import copy


# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Dual_Battery_Ducted_Fan(Propulsor):
    """ Uses two batteries to run a motor connected to a ducted fan; 
        
        Assumptions:
        The primary_battery always runs, while the auxiliary_battery meets additional power needs
        
        Source:
        None
    """       
    def __defaults__(self):
        """ This sets the default values for the network to function.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """            
        self.propulsor            = None
        self.primary_battery      = None # main battery (generally high esp)
        self.auxiliary_battery    = None # used to meet power demands beyond primary
        self.motor_efficiency     = .95
        self.tag                  = 'Network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [newtons]
            results.vehicle_mass_rate   [kg/s]
            conditions.propulsion.primary_battery_draw     [watts]
            conditions.propulsion.primary_battery_energy   [joules]
            conditions.propulsion.auxiliary_battery_draw   [watts]
            conditions.propulsion.auxiliary_battery_energy [joules]
    
            Properties Used:
            Defaulted values
        """         
        
        # unpack

        propulsor          = self.propulsor
        primary_battery    = self.primary_battery
        auxiliary_battery  = self.auxiliary_battery
        conditions         = state.conditions
        numerics           = state.numerics
        
        results=propulsor.evaluate_thrust(state)
     
        Pe     =results.power
        
        try:
            initial_energy           = conditions.propulsion.primary_battery_energy
            initial_energy_auxiliary = conditions.propulsion.auxiliary_battery_energy
            if initial_energy[0][0]==0: #beginning of segment; initialize battery
                primary_battery.current_energy   = primary_battery.current_energy[-1]*np.ones_like(initial_energy)
                auxiliary_battery.current_energy = auxiliary_battery.current_energy[-1]*np.ones_like(initial_energy)
        except AttributeError: #battery energy not initialized, e.g. in takeoff
            primary_battery.current_energy   = np.transpose(np.array([primary_battery.current_energy[-1]*np.ones_like(Pe)]))
            auxiliary_battery.current_energy = np.transpose(np.array([auxiliary_battery.current_energy[-1]*np.ones_like(Pe)]))
       
       
        pbat=-Pe/self.motor_efficiency
        pbat_primary=copy.copy(pbat)                           # prevent deep copy nonsense
        pbat_auxiliary=np.zeros_like(pbat)  
        for i in range(len(pbat)):
            if  pbat[i]<-primary_battery.max_power:            # limit power output of primary_battery
                pbat_primary[i]   = -primary_battery.max_power #-power means discharge
                pbat_auxiliary[i] = pbat[i]-pbat_primary[i]
            elif pbat[i]>primary_battery.max_power:            # limit charging rate of battery
                pbat_primary[i]   = primary_battery.max_power
                pbat_auxiliary[i] = pbat[i]-pbat_primary[i]
            if pbat_primary[i]>0:                              #don't allow non-rechargable battery to charge
                pbat_primary[i]   = 0
                pbat_auxiliary[i] = pbat[i]
     
        primary_battery_logic            = Data()
        primary_battery_logic.power_in   = pbat_primary
        primary_battery_logic.current    = 90.  # use 90 amps as a default for now; will change this for higher fidelity methods
        auxiliary_battery_logic          = copy.copy(primary_battery_logic)
        auxiliary_battery_logic.power_in = pbat_auxiliary
        primary_battery.inputs           = primary_battery_logic
        auxiliary_battery.inputs         = auxiliary_battery_logic
        tol                              = 1e-6
        primary_battery.energy_calc(numerics)
        auxiliary_battery.energy_calc(numerics)
        
        #allow for mass gaining batteries 
        try:
            mdot_primary = find_mass_gain_rate(primary_battery,-(pbat_primary-primary_battery.resistive_losses))
        except AttributeError:
            mdot_primary = np.zeros_like(results.thrust_force_vector[:,0])   
        try:
            mdot_auxiliary = find_mass_gain_rate(auxiliary_battery,-(pbat_auxiliary-auxiliary_battery.resistive_losses))
        except AttributeError:
            mdot_auxiliary = np.zeros_like(results.thrust_force_vector[:,0])
    
        mdot=mdot_primary+mdot_auxiliary
        mdot=np.reshape(mdot, np.shape(conditions.freestream.velocity))
        
        #Pack the conditions for outputs
        primary_battery_draw                 = primary_battery.inputs.power_in
        primary_battery_energy               = primary_battery.current_energy
        auxiliary_battery_draw               = auxiliary_battery.inputs.power_in
        auxiliary_battery_energy             = auxiliary_battery.current_energy
      
        conditions.propulsion.primary_battery_draw   = primary_battery_draw
        conditions.propulsion.primary_battery_energy = primary_battery_energy
        
        conditions.propulsion.auxiliary_battery_draw   = auxiliary_battery_draw
        conditions.propulsion.auxiliary_battery_energy = auxiliary_battery_energy
   
        results.vehicle_mass_rate = mdot

        return results
            
    __call__ = evaluate_thrust
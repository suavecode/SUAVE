## @ingroup Components-Energy-Networks
# Turboelectric_Ducted_Fan.py
#
# Created:  Nov 2019, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Core import Data
# from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate
from SUAVE.Components.Propulsors.Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Turboelectric_Ducted_Fan(Propulsor):
    """ Simply connects a turboelectric power source to a ducted fan, with an assumed ducted fan motor efficiency
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    def __defaults__(self):
        """ This sets the default values for the network to function.
            This network operates slightly different than most as it attaches a propulsor to the net.
    
            Assumptions:
            Your system always uses 90 amps...?
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """         
        
        self.propulsor          = None
        self.motor              = None
        self.powersupply        = None
        # self.motor_efficiency   = .95
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        self.bypass_ratio      = 0.0
        self.areas             = Data()
        self.tag                = 'Network'
    
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
    
            Properties Used:
            Defaulted values
        """         
        
        # unpack

        propulsor       = self.propulsor
        motor           = self.motor
        powersupply     = self.powersupply
    
        conditions  = state.conditions
        numerics    = state.numerics

        # Solve the thrust using the other network (i.e. the ducted fan network)
        results = propulsor.evaluate_thrust(state)

        # Calculate the required electric power to be supplied to the ducted fan motor by dividing the shaft power required by the ducted fan by the efficiency of the ducted fan motor
        powersupply.inputs.power_in = propulsor.thrust.outputs.power/motor.motor_efficiency

        # Calculate the fuel mass flow rate at the turboelectric power supply.
        # This assumes 100% of the electric power delivered by the turboelectric generator is delivered to the motor, i.e. there are no power electronics, avionics, transmission losses, cooling systems, or any other power users.
        mdot = powersupply.energy_calc(conditions, numerics)
        
        results.vehicle_mass_rate   = mdot
        return results
            
    __call__ = evaluate_thrust

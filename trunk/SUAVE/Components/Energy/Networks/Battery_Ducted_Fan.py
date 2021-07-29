## @ingroup Components-Energy-Networks
# Battery_Ducted_Fan.py
#
# Created:  Sep 2014, M. Vegh
# Modified: Jan 2016, T. MacDonald
#           Apr 2019, C. McMillan
#           Apr 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports
import numpy as np
from .Network import Network

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Battery_Ducted_Fan(Network):
    """ Simply connects a battery to a ducted fan, with an assumed motor efficiency
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    def __defaults__(self):
        """ This sets the default values for the network to function.
            This network operates slightly different than most as it attaches a propulsor to the net.
    
            Assumptions:
            
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
        """         

        self.propulsor                 = None
        self.battery                   = None
        self.motor_efficiency          = 0.0
        self.tag                       = 'Battery_Ducted_Fan'
        self.number_of_engines         = 0.
        self.nacelle_diameter          = 0.
        self.esc                       = None
        self.avionics                  = None
        self.payload                   = None
        self.voltage                   = None
        self.tag                       = 'Network'
        self.generative_design_minimum = 0

    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            Constant mass batteries
            ESC input voltage is constant at max battery voltage
            
    
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
        conditions = state.conditions
        numerics   = state.numerics
        esc        = self.esc
        avionics   = self.avionics
        payload    = self.payload 
        battery    = self.battery
        propulsor  = self.propulsor
        battery    = self.battery

        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy

        # Calculate ducted fan power
        results             = propulsor.evaluate_thrust(state)
        propulsive_power    = np.reshape(results.power, (-1,1))
        motor_power         = propulsive_power/self.motor_efficiency 

        # Run the ESC
        esc.inputs.voltagein    = self.voltage
        esc.voltageout(conditions)
        esc.inputs.currentout   = motor_power/esc.outputs.voltageout
        esc.currentin(conditions)
        esc_power               = esc.inputs.voltagein*esc.outputs.currentin
        
        # Run the avionics
        avionics.power()

        # Run the payload
        payload.power()

        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power

        # Calculate avionics and payload current
        avionics_payload_current = avionics_payload_power/self.voltage

        # link to the battery
        battery.inputs.current  = esc.outputs.currentin + avionics_payload_current 
        battery.inputs.power_in = -(esc_power + avionics_payload_power)
        battery.energy_calc(numerics)        
    
        # No mass gaining batteries
        mdot = np.zeros(np.shape(conditions.freestream.velocity))

        # Pack the conditions for outputs
        current              = esc.outputs.currentin
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy
        voltage_open_circuit = battery.voltage_open_circuit
          
        conditions.propulsion.current                      = current
        conditions.propulsion.battery_draw                 = battery_draw
        conditions.propulsion.battery_energy               = battery_energy
        conditions.propulsion.battery_voltage_open_circuit = voltage_open_circuit
        
        results.vehicle_mass_rate   = mdot
        return results
            
    __call__ = evaluate_thrust

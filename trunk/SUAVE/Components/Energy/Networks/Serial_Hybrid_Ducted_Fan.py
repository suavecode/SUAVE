## @ingroup Components-Energy-Networks
# Battery_Ducted_Fan.py
#
# Created:  Sep 2014, M. Vegh
# Modified: Jan 2016, T. MacDonaldb

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Core import Data, Units
from SUAVE.Methods.Power.Battery.Variable_Mass import find_mass_gain_rate
from SUAVE.Components.Propulsors.Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Serial_Hybrid_Ducted_Fan(Propulsor):
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
        
        self.propulsor        = None
        self.battery          = None
        self.motor_efficiency = .95 
        self.esc              = None
        self.avionics         = None
        self.payload          = None
        self.voltage          = None
        self.generator        = None
        self.tag              = 'Network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            Doesn't allow for mass gaining batteries
    
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
         
        if state.residuals.forces[0,0] < 1e-7 and state.residuals.forces[0,0] > 0:
            a123= 1

        #Cameron's attempt
        # unpack
        conditions = state.conditions
        numerics   = state.numerics
        esc        = self.esc
        avionics   = self.avionics
        payload    = self.payload 
        battery    = self.battery
        propulsor  = self.propulsor
        battery    = self.battery
        generator  = self.generator
        

        range_extender_power = 500000 #Watts
        range_extender_sfc = (.3/3600) #kg/(kW*s)
        
        power_generated = np.ones_like(conditions.propulsion.battery_energy)*range_extender_power
        mdot = range_extender_sfc * (power_generated/1000) 
        
        generator.calculate_power(conditions)

        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy

        # Step 0 ducted fan power
        results             = propulsor.evaluate_thrust(state)
        propulsive_power    = results.power
        motor_power         = propulsive_power/self.motor_efficiency 
      
        #battery.inputs = battery_logic
        tol = 1e-6  

        # Set the esc input voltage
        esc.inputs.voltagein = self.voltage

        # Calculate the esc output voltage
        esc.voltageout(conditions)

        # Run the avionics
        avionics.power()

        # Run the payload
        payload.power()

        # Calculate the esc input current
        esc.inputs.currentout =  np.transpose(motor_power/np.transpose(esc.outputs.voltageout))
        
        # Run the esc
        esc.currentin(conditions)

        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power

        # Calculate avionics and payload current
        avionics_payload_current = avionics_payload_power/self.voltage

        # link
        battery.inputs.current  = esc.outputs.currentin + avionics_payload_current
        battery.inputs.power_in = -((esc.inputs.voltagein)*esc.outputs.currentin + avionics_payload_power) + (
                generator.outputs.power_generated)
        
        # Run the battery
        battery.energy_calc(numerics)        
    
        


        #try:
        #    initial_energy = conditions.propulsion.battery_energy
        #    if initial_energy[0][0]==0: #beginning of segment; initialize battery
        #        battery.current_energy = battery.current_energy[-1]*np.ones_like(initial_energy)
        #except AttributeError: #battery energy not initialized, e.g. in takeoff
        #    battery.current_energy=np.transpose(np.array([battery.current_energy[-1]*np.ones_like(Pe)]))
        
        
    


        #mdot = np.zeros(np.shape(conditions.freestream.velocity))

        # Pack the conditions for outputs
        current              = esc.outputs.currentin
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy
        voltage_open_circuit = battery.voltage_open_circuit
          
        conditions.propulsion.current              = current
        conditions.propulsion.battery_draw         = battery_draw
        conditions.propulsion.battery_energy       = battery_energy
        conditions.propulsion.voltage_open_circuit = voltage_open_circuit
        
        results.vehicle_mass_rate   = generator.outputs.vehicle_mass_rate
        return results
            
    __call__ = evaluate_thrust

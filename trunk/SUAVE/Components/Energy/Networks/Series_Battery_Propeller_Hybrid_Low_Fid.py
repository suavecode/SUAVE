# Series_Battery_Propller_Hybrid.py
# 
# Created:  Jul 2015, E. Botero
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Propulsors.Propulsor import Propulsor

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Series_Battery_Propeller_Hybrid_Low_Fid(Propulsor):
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
        self.combustion_engine = None
        self.generator         = None
        self.thrust_angle      = 0.0
        self.tag               = 'network'
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
    
        # unpack
        conditions = state.conditions
        numerics   = state.numerics
        motor      = self.motor
        propeller  = self.propeller
        esc        = self.esc
        avionics   = self.avionics
        payload    = self.payload
        battery    = self.battery
        engine     = self.combustion_engine
        generator  = self.generator
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy  
    
        # Step 1 battery power
        esc.inputs.voltagein = self.voltage
        # Step 2
        esc.voltageout(conditions)
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        motor.power_lo(conditions)
        #link
        propeller.inputs.power = motor.outputs.power
        F,P = propeller.spin_lo(conditions)
        
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta        = conditions.propulsion.throttle[:,0,None]
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]

        # Run the avionics
        avionics.power()

        # Run the payload
        payload.power()
        

        # link
        esc.inputs.currentout =  motor.outputs.current
        
        # Run the esc
        esc.currentin()

        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power

        # Calculate avionics and payload current
        avionics_payload_current = avionics_payload_power/self.voltage
        
        # Run the internal combustion engine
        engine.power(conditions)

        # Assume the generators voltage can be normalized:
        Pgen = engine.outputs.power * generator.motor_efficiency
        
        # Make sure the power generated is always positve:
        Pgen[Pgen<0.] = 0.
        
        # Now the normalized current
        i_gen = Pgen/self.voltage
        

        # link
        battery.inputs.current  = esc.outputs.currentin*self.number_of_engines + avionics_payload_current-i_gen
        battery.inputs.power_in = -(esc.outputs.voltageout*esc.outputs.currentin*self.number_of_engines + avionics_payload_power-Pgen)
        battery.energy_calc(numerics)        
    
        # Pack the conditions for outputs
        current              = esc.outputs.currentin*self.number_of_engines
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy

        conditions.propulsion.current              = battery.inputs.current
        conditions.propulsion.battery_draw         = battery_draw
        conditions.propulsion.battery_energy       = battery_energy
        conditions.propulsion.voltage_open_circuit = self.voltage
        conditions.propulsion.generator_power      = Pgen
        conditions.propulsion.power                = -(esc.outputs.voltageout*esc.outputs.currentin*self.number_of_engines + avionics_payload_power)
        
        # Create the outputs
        F    = self.number_of_engines * F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]      
        mdot = engine.outputs.fuel_flow_rate
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
            
        return results
            
    __call__ = evaluate_thrust

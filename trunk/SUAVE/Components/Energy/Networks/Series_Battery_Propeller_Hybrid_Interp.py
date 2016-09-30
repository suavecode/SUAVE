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

from SUAVE.Core import Data, Units

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------
class Series_Battery_Propeller_Hybrid_Interp(Propulsor):
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
        self.max_omega         = 0.0
        self.motors_per_prop   = None
        self.number_of_props   = None
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
        n_motor    = self.motors_per_prop
        n_props    = self.number_of_props
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy  

        
        # Throttle the system
        ## Check to see if magic thrust is needed
        #eta          = conditions.propulsion.throttle[:,0,None] * 1.0
        #eta_c        = conditions.propulsion.throttle[:,0,None] * 1.0
        #eta_c[eta_c>1.0] = 1.0
        #eta_c[eta_c<0.0] = 0.0
        #omega        = self.max_omega * eta_c

        ## link
        #propeller.inputs.omega = omega
        #propeller.thrust_angle = self.thrust_angle
        
        ## Run the propeller
        #F, Q, P, Cp = propeller.spin(conditions)   
        
        #P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        #F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
        ##Q[eta>1.0] = Q[eta>1.0]*eta[eta>1.0]
        #P[eta<0.0] = P[eta<0.0]*eta[eta<0.0]
        #F[eta<0.0] = F[eta<0.0]*eta[eta<0.0]    
        ##Q[eta<0.0] = Q[eta<0.0]*eta[eta<0.0] 
        
        # Throttle the system
        # Check to see if magic thrust is needed
        eta          = conditions.propulsion.throttle[:,0,None] * 1.0
        eta_c        = conditions.propulsion.throttle[:,0,None] * 1.0
        
        eta_c[eta_c<=-np.pi/2] = -np.pi/2
        eta_c[eta_c>=np.pi/2]  =  np.pi/2

        
        omega = conditions.propulsion.rpm
        conditions.propulsion.pitch_command =  eta_c
        #link
        propeller.inputs.omega = omega
        propeller.thrust_angle = self.thrust_angle
        
        # Run the propeller
        F, Q, P, Cp = propeller.spin_variable_pitch(conditions)   
        #F, Q, P, Cp = propeller.spin(conditions)   
        #P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        #F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
        ##Q[eta>1.0] = Q[eta>1.0]*eta[eta>1.0]
        #P[eta<0.0] = P[eta<0.0]*eta[eta<0.0]
        #F[eta<0.0] = F[eta<0.0]*eta[eta<0.0]    
        ##Q[eta<0.0] = Q[eta<0.0]*eta[eta<0.0] 
        
        
        
        # Run the motor
        motor.inputs.omega  = omega
        motor.inputs.torque = Q/n_motor
        motor.power_from_fit()
        
        # Calculate the current going into the motor and ESC
        esc.outputs.currentin = n_motor*motor.outputs.power_in/self.voltage

        # Run the avionics
        avionics.power()

        # Run the payload
        payload.power()

        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power

        # Calculate avionics and payload current
        avionics_payload_current = avionics_payload_power/self.voltage
        
        # Run the internal combustion engine
        engine.power(conditions)
        
        # Link the internal combustion engine to the generator
        generator.inputs       = engine.outputs
        generator.inputs.omega = engine.speed
        
        # Run the generator
        generator.voltage_current(conditions)
        
        # Assume the generators voltage can be normalized:
        Pgen = generator.outputs.voltage * generator.outputs.current
        
        # Make sure the power generated is always positve:
        Pgen[Pgen<0.] = 0.
        
        # Now the normalized current
        i_gen = Pgen/self.voltage
        
        # link
        battery.inputs.current  = esc.outputs.currentin*n_props + avionics_payload_current-i_gen
        battery.inputs.power_in = -(motor.outputs.power_in*n_props + avionics_payload_power-Pgen)
        battery.energy_calc(numerics)        
    
        # Pack the conditions for outputs
        rpm                  = omega / Units.rpm
        current              = esc.outputs.currentin
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy
        voltage_open_circuit = battery.voltage_open_circuit
        voltage_under_load   = battery.voltage_under_load    
          
        #conditions.propulsion.rpm                  = rpm
        conditions.propulsion.current              = current
        conditions.propulsion.battery_draw         = battery_draw
        conditions.propulsion.battery_energy       = battery_energy
        conditions.propulsion.voltage_open_circuit = voltage_open_circuit
        conditions.propulsion.voltage_under_load   = voltage_under_load  
        conditions.propulsion.motor_torque         = Q/n_motor
        conditions.propulsion.propeller_torque     = Q
        
        # Create the outputs
        F    = self.number_of_engines * F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]      
        mdot = engine.outputs.fuel_flow_rate
        results = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot
            
        return results
    
    
    def unpack_unknowns(self,segment,state):
        """"""        
        
        # Here we are going to unpack the unknowns (Cp) provided for this network
        #state.conditions.propulsion.propeller_power_coefficient = state.unknowns.propeller_power_coefficient
        #state.conditions.propulsion.battery_voltage_under_load  = state.unknowns.battery_voltage_under_load
        
        return
    
    def residuals(self,segment,state):
        """"""        
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        
        # Unpack
        #q_motor   = state.conditions.propulsion.motor_torque
        #q_prop    = state.conditions.propulsion.propeller_torque
        #v_actual  = state.conditions.propulsion.voltage_under_load
        #v_predict = state.unknowns.battery_voltage_under_load
        #v_max     = self.voltage
        
        # Return the residuals
        
        # HARD CODED 2!!!!!!!!!!!!!!
        #state.residuals.network[:,0] = 2.*q_motor[:,0] - q_prop[:,0]
        # HARD CODED 2!!!!!!!!!!!!!!
        
        #state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max
        
        return    
            
    __call__ = evaluate_thrust

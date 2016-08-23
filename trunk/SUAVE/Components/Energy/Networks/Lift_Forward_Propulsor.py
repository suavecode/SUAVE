# Lift_Forward_propulsor.py
# 
# Created: Jan 2016, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
import datetime
import time
from SUAVE.Core import Units
from SUAVE.Components.Propulsors.Propulsor import Propulsor
from SUAVE.Components.Energy.Networks import Battery_Propeller

from SUAVE.Core import (
Data, Container
)

# We have two inputs, the forward throttle and the lift throttle setting

# Since this is an airplane first and foremost, the "throttle" will be for forward thrust
# The new unknown will be for lift throttle, because of the assumption on throttle something needs to be done...

# Want only 1 residual on voltage

# For any segment using this, body angle can't be an unknown.


class Lift_Forward(Propulsor):
    def __defaults__(self):
        self.motor_lift                = None
        self.motor_forward             = None
        self.propeller_lift            = None
        self.propeller_forward         = None
        self.esc_lift                  = None
        self.esc_forward               = None
        self.avionics                  = None
        self.payload                   = None
        self.battery                   = None
        self.nacelle_diameter_lift     = None
        self.nacelle_diameter_forward  = None
        self.engine_length_lift        = None
        self.engine_length_forward     = None
        self.number_of_engines_lift    = None
        self.number_of_engines_forward = None
        self.voltage                   = None
        self.thrust_angle_lift         = 0.0
        self.thrust_angle_forward      = 0.0
        
        pass
        
    def evaluate_thrust(self,state):
        
        # unpack
        conditions        = state.conditions
        numerics          = state.numerics
        motor_lift        = self.motor_lift 
        motor_forward     = self.motor_forward
        propeller_lift    = self.propeller_lift 
        propeller_forward = self.propeller_forward
        esc_lift          = self.esc_lift
        esc_forward       = self.esc_forward        
        avionics          = self.avionics
        payload           = self.payload
        battery           = self.battery
        num_lift          = self.number_of_engines_lift
        num_forward       = self.number_of_engines_forward
        
        ###
        # Setup batteries and ESC's
        ###
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy    
        
        volts = state.unknowns.battery_voltage_under_load   
        volts[volts>self.voltage] = self.voltage
        
        # ESC Voltage
        esc_lift.inputs.voltagein    = volts      
        esc_forward.inputs.voltagein = volts 
        
        ###
        # Evaluate thrust from the forward propulsors
        ###
        
        # Throttle the voltage
        esc_forward.voltageout(conditions)       
        # link
        motor_forward.inputs.voltage = esc_forward.outputs.voltageout
        
        # Run the motor
        motor_forward.omega(conditions)
        # link
        propeller_forward.inputs.omega =  motor_forward.outputs.omega
        propeller_forward.thrust_angle = self.thrust_angle_forward   
        
        # Run the propeller
        F_forward, Q_forward, P_forward, Cp_forward = propeller_forward.spin(conditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = conditions.propulsion.throttle[:,0,None]
        P_forward[eta>1.0] = P_forward[eta>1.0]*eta[eta>1.0]
        F_forward[eta>1.0] = F_forward[eta>1.0]*eta[eta>1.0]        
        
        # Run the motor for current
        motor_forward.current(conditions)  
        # link
        esc_forward.inputs.currentout =  motor_forward.outputs.current     
        
        # Run the esc
        esc_forward.currentin()        
       
        ###
        # Evaluate thrust from the lift propulsors
        ###
        
        # Make a new set of konditions, since there are differences for the esc and motor
        konditions                 = Data()
        konditions.propulsion      = Data()
        konditions.freestream      = Data()
        konditions.frames          = Data()
        konditions.frames.inertial = Data()
        konditions.frames.body     = Data()
        konditions.propulsion.throttle                    = conditions.propulsion.lift_throttle * 1.
        konditions.propulsion.propeller_power_coefficient = conditions.propulsion.propeller_power_coefficient_lift * 1.
        konditions.freestream.density                     = conditions.freestream.density * 1.
        konditions.freestream.velocity                    = conditions.freestream.velocity * 1.
        konditions.freestream.dynamic_viscosity           = conditions.freestream.dynamic_viscosity * 1.
        konditions.freestream.speed_of_sound              = conditions.freestream.speed_of_sound *1.
        konditions.freestream.temperature                 = conditions.freestream.temperature * 1.
        konditions.frames.inertial.velocity_vector        = conditions.frames.inertial.velocity_vector *1.
        konditions.frames.body.transform_to_inertial      = conditions.frames.body.transform_to_inertial
        
        # Throttle the voltage
        esc_lift.voltageout(konditions)       
        # link
        motor_lift.inputs.voltage = esc_lift.outputs.voltageout
        
        # Run the motor
        motor_lift.omega(konditions)
        # link
        propeller_lift.inputs.omega =  motor_lift.outputs.omega
        propeller_lift.thrust_angle = self.thrust_angle_lift
        
        # Run the propeller
        F_lift, Q_lift, P_lift, Cp_lift = propeller_lift.spin(konditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = state.conditions.propulsion.lift_throttle
        P_lift[eta>1.0] = P_lift[eta>1.0]*eta[eta>1.0]
        F_lift[eta>1.0] = F_lift[eta>1.0]*eta[eta>1.0]        
        
        # Run the motor for current
        motor_lift.current(conditions)  
        # link
        esc_lift.inputs.currentout =  motor_lift.outputs.current     
        
        # Run the esc
        esc_lift.currentin()          
        
        ###
        # Combine the thrusts and powers
        ###
        
        # Run the avionics
        avionics.power()
    
        # Run the payload
        payload.power()
        
        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power
    
        # Calculate avionics and payload current
        i_avionics_payload = avionics_payload_power/state.unknowns.battery_voltage_under_load    
        
        # Add up the power usages
        i_lift    = esc_lift.outputs.currentin*num_lift 
        i_forward = esc_forward.outputs.currentin*num_forward
        
        current_total = i_lift + i_forward + i_avionics_payload
        power_total   = current_total * state.unknowns.battery_voltage_under_load  
        
        battery.inputs.current  = current_total
        battery.inputs.power_in = - power_total
        
        # Run the battery
        battery.energy_calc(numerics)   
        
        # Pack the conditions
        rpm_lift             = motor_lift.outputs.omega*60./(2.*np.pi)
        rpm_forward          = motor_forward.outputs.omega*60./(2.*np.pi)        
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy
        voltage_open_circuit = battery.voltage_open_circuit
        voltage_under_load   = battery.voltage_under_load    
    
        conditions.propulsion.rpm_lift                 = rpm_lift
        conditions.propulsion.rpm_forward              = rpm_forward
        conditions.propulsion.current_lift             = i_lift 
        conditions.propulsion.current_forward          = i_forward 
        conditions.propulsion.motor_torque_lift        = motor_lift.outputs.torque
        conditions.propulsion.motor_torque_forward     = motor_forward.outputs.torque
        conditions.propulsion.propeller_torque_lift    = Q_lift   
        conditions.propulsion.propeller_torque_forward = Q_forward       
          
        conditions.propulsion.battery_draw             = battery_draw
        conditions.propulsion.battery_energy           = battery_energy
        conditions.propulsion.voltage_open_circuit     = voltage_open_circuit
        conditions.propulsion.voltage_under_load       = voltage_under_load      
        
        # Calculate the thrust and mdot
        F_lift_total    = F_lift*num_lift * [np.cos(self.thrust_angle_lift),0,-np.sin(self.thrust_angle_lift)]    
        F_forward_total = F_forward*num_forward * [np.cos(self.thrust_angle_forward),0,-np.sin(self.thrust_angle_forward)] 
       
        F_total = F_lift_total + F_forward_total
        mdot    = np.zeros_like(F_total)
        
        results = Data()
        results.thrust_force_vector = F_total
        results.vehicle_mass_rate   = mdot
        
        return results
    
    def unpack_unknowns(self,segment,state):
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        state.conditions.propulsion.lift_throttle                    = state.unknowns.lift_throttle
        state.conditions.propulsion.battery_voltage_under_load       = state.unknowns.battery_voltage_under_load
        state.conditions.propulsion.propeller_power_coefficient      = state.unknowns.propeller_power_coefficient
        state.conditions.propulsion.propeller_power_coefficient_lift = state.unknowns.propeller_power_coefficient_lift
        state.conditions.propulsion.throttle                         = state.unknowns.throttle
        
        return
    
    
    def unpack_unknowns_no_lift(self,segment,state):
        
        ones = state.ones_row
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        state.conditions.propulsion.lift_throttle                    = 0.0 * ones(1)
        state.conditions.propulsion.battery_voltage_under_load       = state.unknowns.battery_voltage_under_load
        state.conditions.propulsion.propeller_power_coefficient      = state.unknowns.propeller_power_coefficient
        state.conditions.propulsion.propeller_power_coefficient_lift = 0.0 * ones(1)
        state.conditions.propulsion.throttle                         = state.unknowns.throttle
        
        return    
    
    def unpack_unknowns_no_forward(self,segment,state):
        
        ones = state.ones_row
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        state.conditions.propulsion.lift_throttle                    = state.unknowns.lift_throttle
        state.conditions.propulsion.battery_voltage_under_load       = state.unknowns.battery_voltage_under_load
        state.conditions.propulsion.propeller_power_coefficient      = 0.0 * ones(1)
        state.conditions.propulsion.propeller_power_coefficient_lift = state.unknowns.propeller_power_coefficient_lift
        state.conditions.propulsion.throttle                         = 0.0 * ones(1)
        
        return    
    
    
    def residuals(self,segment,state):
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        q_motor_forward = state.conditions.propulsion.motor_torque_forward
        q_prop_forward  = state.conditions.propulsion.propeller_torque_forward
        q_motor_lift    = state.conditions.propulsion.motor_torque_lift
        q_prop_lift     = state.conditions.propulsion.propeller_torque_lift        
        
        v_actual        = state.conditions.propulsion.voltage_under_load
        v_predict       = state.unknowns.battery_voltage_under_load
        v_max           = self.voltage        
        
        # Return the residuals
        state.residuals.network[:,0] = (q_motor_forward[:,0] - q_prop_forward[:,0])/q_motor_forward[:,0] 
        state.residuals.network[:,1] = (q_motor_lift[:,0] - q_prop_lift[:,0])/q_motor_lift[:,0]
        state.residuals.network[:,2] = (v_predict[:,0] - v_actual[:,0])/v_max  
        
        return
    
    
    def residuals_no_lift(self,segment,state):
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        q_motor_forward = state.conditions.propulsion.motor_torque_forward
        q_prop_forward  = state.conditions.propulsion.propeller_torque_forward   
        
        v_actual        = state.conditions.propulsion.voltage_under_load
        v_predict       = state.unknowns.battery_voltage_under_load
        v_max           = self.voltage        
        
        # Return the residuals
        state.residuals.network[:,0] = q_motor_forward[:,0] - q_prop_forward[:,0]
        state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max  
        
        return    
    
    def residuals_no_forward(self,segment,state):
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        q_motor_lift    = state.conditions.propulsion.motor_torque_lift
        q_prop_lift     = state.conditions.propulsion.propeller_torque_lift        
        
        v_actual        = state.conditions.propulsion.voltage_under_load
        v_predict       = state.unknowns.battery_voltage_under_load
        v_max           = self.voltage        
        
        # Return the residuals
        state.residuals.network[:,0] = (q_motor_lift[:,0] - q_prop_lift[:,0])/q_motor_lift[:,0]
        state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max  
        
        return

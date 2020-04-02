# Lift_Cruise_Low_Fidelity.py
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


class Lift_Cruise_Low_Fidelity(Propulsor):
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
        self.areas = Data()
        
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
        
        ##
        # SETUP BATTERIES AND ESC's
        ##
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy    
        
        #volts = state.unknowns.battery_voltage_under_load * 1. 
        #volts[volts>self.voltage] = self.voltage
        volts = self.voltage
        
        # ESC Voltage
        esc_lift.inputs.voltagein    = volts      
        esc_forward.inputs.voltagein = volts 
        
        ##
        # EVALUATE THRUST FROM FORWARD PROPULSORS 
        ##
        
        # Throttle the voltage
        esc_forward.voltageout(conditions)       
        # link
        motor_forward.inputs.voltage = esc_forward.outputs.voltageout
        
        motor_forward.power_lo(conditions)
        #link
        propeller_forward.inputs.power = motor_forward.outputs.power
        F_forward,P_forward = propeller_forward.spin_lo(conditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = conditions.propulsion.throttle[:,0,None]
        P_forward[eta>1.0] = P_forward[eta>1.0]*eta[eta>1.0]
        F_forward[eta>1.0] = F_forward[eta>1.0]*eta[eta>1.0]        
        
        ## Run the motor for current
        #motor_forward.current(conditions)  
        
        # link
        esc_forward.inputs.currentout =  motor_forward.outputs.current     
        
        # Run the esc
        esc_forward.currentin()        
       
        ## 
        # EVALUATE THRUST FROM LIFT PROPULSORS 
        ## 
        
        # Make a new set of konditions, since there are differences for the esc and motor
        konditions                 = Data()
        konditions.propulsion      = Data()
        konditions.freestream      = Data()
        konditions.frames          = Data()
        konditions.frames.inertial = Data()
        konditions.frames.body     = Data()
        konditions.propulsion.throttle                    = conditions.propulsion.throttle_lift* 1.
        konditions.freestream.density                     = conditions.freestream.density * 1.
        konditions.freestream.velocity                    = conditions.freestream.velocity * 1.
        konditions.freestream.dynamic_viscosity           = conditions.freestream.dynamic_viscosity * 1.
        konditions.freestream.speed_of_sound              = conditions.freestream.speed_of_sound *1.
        konditions.freestream.temperature                 = conditions.freestream.temperature * 1.
        konditions.freestream.altitude                    = conditions.freestream.altitude * 1.
        konditions.frames.inertial.velocity_vector        = conditions.frames.inertial.velocity_vector *1.
        konditions.frames.body.transform_to_inertial      = conditions.frames.body.transform_to_inertial
        
        # Throttle the voltage
        esc_lift.voltageout(konditions)       
        # link
        motor_lift.inputs.voltage = esc_lift.outputs.voltageout
        
        motor_lift.power_lo(conditions)
        #link
        propeller_lift.inputs.power = motor_lift.outputs.power
        F_lift,P_lift = propeller_lift.spin_lo(conditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = state.conditions.propulsion.throttle_lift
        P_lift[eta>1.0] = P_lift[eta>1.0]*eta[eta>1.0]
        F_lift[eta>1.0] = F_lift[eta>1.0]*eta[eta>1.0]        
        
        ## Run the motor for current
        #motor_lift.current(conditions)
        
        # link
        esc_lift.inputs.currentout =  motor_lift.outputs.current     
        
        # Run the esc
        esc_lift.currentin()          
        
        ##
        # COMBINE THRUST AND POWER
        ##
        
        # Run the avionics
        avionics.power()
    
        # Run the payload
        payload.power()
        
        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power
    
        # Calculate avionics and payload current
        i_avionics_payload = avionics_payload_power/volts   
        
        # Add up the power usages
        i_lift    = esc_lift.outputs.currentin*num_lift 
        i_forward = esc_forward.outputs.currentin*num_forward
        
        current_total = i_lift + i_forward + i_avionics_payload
        power_total   = current_total * volts   
        
        battery.inputs.current  = current_total
        battery.inputs.power_in = - power_total
        
        # Run the battery
        battery.energy_discharge(numerics)   
        
        # Pack the conditions     
        conditions.propulsion.current_lift                 = i_lift 
        conditions.propulsion.current_forward              = i_forward  
        conditions.propulsion.battery_draw                 = battery.inputs.power_in 
        conditions.propulsion.battery_energy               = battery.current_energy 
        conditions.propulsion.battery_voltage_open_circuit =  battery.voltage_open_circuit
        conditions.propulsion.battery_voltage_under_load   =  battery.voltage_under_load       
        
        # Calculate the thrust and mdot
        F_lift_total    = F_lift*num_lift * [np.cos(self.thrust_angle_lift),0,-np.sin(self.thrust_angle_lift)]    
        F_forward_total = F_forward*num_forward * [np.cos(self.thrust_angle_forward),0,-np.sin(self.thrust_angle_forward)] 
       
        F_total = F_lift_total + F_forward_total
        mdot    = np.zeros_like(F_total)
        
        results = Data()
        results.thrust_force_vector = F_total
        results.vehicle_mass_rate   = mdot
        
        return results
    
    def unpack_unknowns_transition(self,segment):
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        segment.state.conditions.propulsion.throttle_lift                   = segment.state.unknowns.throttle_lift
        segment.state.conditions.propulsion.throttle                         = segment.state.unknowns.throttle
        
        return
    
    
    def unpack_unknowns_no_lift(self,segment):
        
        ones = segment.state.ones_row
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        segment.state.conditions.propulsion.throttle_lift                   = 0.0 * ones(1)
        segment.state.conditions.propulsion.throttle                         = segment.state.unknowns.throttle
        
        return    
    
    def unpack_unknowns_no_forward(self,segment):
        
        ones = segment.state.ones_row
        
        # Here we are going to unpack the unknowns (Cps,throttlevoltage) provided for this network
        segment.state.conditions.propulsion.throttle_lift            = segment.state.unknowns.throttle
        segment.state.conditions.propulsion.throttle                         = 0.0 * ones(1)
        
        return    
    
    
    def residuals(self,segment):
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        #q_motor_forward = segment.state.conditions.propulsion.motor_torque_forward
        #q_prop_forward  = segment.state.conditions.propulsion.propeller_torque_forward
        #q_motor_lift    = segment.state.conditions.propulsion.motor_torque_lift
        #q_prop_lift     = segment.state.conditions.propulsion.propeller_torque_lift        
        
        #v_actual        = state.conditions.propulsion.battery_voltage_under_load 
        #v_predict       = state.unknowns.battery_voltage_under_load
        #v_max           = self.voltage        
        
        # Return the residuals
        #state.residuals.network[:,0] = (q_motor_forward[:,0] - q_prop_forward[:,0])/q_motor_forward[:,0] 
        #state.residuals.network[:,1] = (q_motor_lift[:,0] - q_prop_lift[:,0])/q_motor_lift[:,0]
        #state.residuals.network[:,2] = (v_predict[:,0] - v_actual[:,0])/v_max  
        
        return
    
    
    def residuals_no_lift(self,segment):
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        #q_motor_forward = state.conditions.propulsion.motor_torque_forward
        #q_prop_forward  = state.conditions.propulsion.propeller_torque_forward   
        
        #v_actual        = state.conditions.propulsion.battery_voltage_under_load 
        #v_predict       = state.unknowns.battery_voltage_under_load
        #v_max           = self.voltage        
        
        # Return the residuals
        #state.residuals.network[:,0] = q_motor_forward[:,0] - q_prop_forward[:,0]
        #state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max  
        
        return    
    
    def residuals_no_forward(self,segment):
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        #q_motor_lift    = state.conditions.propulsion.motor_torque_lift
        #q_prop_lift     = state.conditions.propulsion.propeller_torque_lift        
        
        #v_actual        = state.conditions.propulsion.battery_voltage_under_load 
        #v_predict       = state.unknowns.battery_voltage_under_load
        #v_max           = self.voltage        
        
        # Return the residuals
        #state.residuals.network[:,0] = (q_motor_lift[:,0] - q_prop_lift[:,0])/q_motor_lift[:,0]
        #state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max  
        
        return
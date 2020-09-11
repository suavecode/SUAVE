## @ingroup Components-Energy-Networks
# Vectored_Thrust.py
# 
# Created:  Nov 2018, M.Clarke
#           Mar 2020, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Components.Propulsors.Propulsor import Propulsor
import math 
from SUAVE.Core import  Units, Data

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Vectored_Thrust(Propulsor):
    """ This is a simple network with a battery powering a rotor through
        an electric motor

        Assumptions:
        None
        
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
        self.tag                      = 'vectored_thrust'
        self.motor                    = None
        self.rotor                    = None
        self.esc                      = None
        self.avionics                 = None
        self.payload                  = None
        self.battery                  = None
        self.nacelle_diameter         = None
        self.engine_length            = None
        self.number_of_engines        = None
        self.voltage                  = None
        self.thrust_angle             = 0.0 
        self.pitch_command            = 0.0 
        self.thrust_angle_start       = None
        self.thrust_angle_end         = None        
    
    # manage process with a driver function
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            Caps the throttle at 110% and linearly interpolates thrust off that
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [newtons]
            results.vehicle_mass_rate   [kg/s]
            conditions.propulsion:
                rpm                  [radians/sec]
                current              [amps]
                battery_draw         [watts]
                battery_energy       [joules]
                voltage_open_circuit [volts]
                voltage_under_load   [volts]
                motor_torque         [N-M]
                propeller_torque     [N-M]
    
            Properties Used:
            Defaulted values
        """          
    
        # unpack
        conditions  = state.conditions
        numerics    = state.numerics
        motor       = self.motor
        rotor       = self.rotor
        esc         = self.esc
        avionics    = self.avionics
        payload     = self.payload
        battery     = self.battery
        num_engines = self.number_of_engines
        t_nondim    = state.numerics.dimensionless.control_points
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy  
        
        volts = state.unknowns.battery_voltage_under_load * 1. 
        volts[volts>self.voltage] = self.voltage 
        
        # ESC Voltage
        esc.inputs.voltagein = volts
        
        #---------------------------------------------------------------
        # EVALUATE THRUST FROM PROPULSORS 
        #---------------------------------------------------------------        
        # Step 2
        esc.voltageout(conditions)  
        
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        
        # Run the motor
        motor.omega(conditions)
        
        # Define the thrust angle 
        thrust_angle = self.thrust_angle
                
        # link
        rotor.inputs.omega  = motor.outputs.omega
        rotor.thrust_angle  = thrust_angle
        rotor.pitch_command = self.pitch_command 
        
        # Run the rotor     
        F, Q, P, Cp , outputs, etap = rotor.spin(conditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta        = conditions.propulsion.throttle[:,0,None]
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]

        # Run the avionics
        avionics.power()

        # Run the payload
        payload.power()
        
        # Run the motor for current
        i, etam = motor.current(conditions)  
        
        # Fix the current for the throttle cap
        motor.outputs.current[eta>1.0] = motor.outputs.current[eta>1.0]*eta[eta>1.0]
         
        # link
        esc.inputs.currentout =  motor.outputs.current
        
        # Run the esc
        esc.currentin(conditions)

        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power

        # Calculate avionics and payload current
        avionics_payload_current = avionics_payload_power/self.voltage

        # link
        propeller_current       = esc.outputs.currentin*num_engines
        total_current           = propeller_current + avionics_payload_current
        battery.inputs.current  = total_current 
        battery.inputs.power_in = -(esc.outputs.voltageout*esc.outputs.currentin*num_engines + avionics_payload_power)
        battery.energy_calc(numerics)        
        
        # Pack the conditions for outputs
        rpm                  = motor.outputs.omega / Units.rpm
        a                    = conditions.freestream.speed_of_sound
        R                    = rotor.tip_radius       
        current              = esc.outputs.currentin
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy
        voltage_open_circuit = battery.voltage_open_circuit
        voltage_under_load   = battery.voltage_under_load    
          
        conditions.propulsion.rpm                             = rpm
        conditions.propulsion.current                         = current
        conditions.propulsion.battery_draw                    = battery_draw
        conditions.propulsion.battery_energy                  = battery_energy 
        conditions.propulsion.voltage_open_circuit            = voltage_open_circuit
        conditions.propulsion.voltage_under_load              = voltage_under_load  
        conditions.propulsion.motor_torque                    = motor.outputs.torque
        conditions.propulsion.propeller_torque                = Q
        conditions.propulsion.motor_efficiency                = etam
        conditions.propulsion.acoustic_outputs[rotor.tag]     = outputs
        conditions.propulsion.battery_specfic_power           = -battery_draw/battery.mass_properties.mass #Wh/kg
        conditions.propulsion.electronics_efficiency          = -(P*num_engines)/battery_draw   
        conditions.propulsion.propeller_tip_mach              = (R*rpm*Units.rpm)/a
        conditions.propulsion.battery_current                 = total_current
        conditions.propulsion.battery_efficiency              = (battery_draw+battery.resistive_losses)/battery_draw
        conditions.propulsion.payload_efficiency              = (battery_draw+(avionics.outputs.power + payload.outputs.power))/battery_draw            
        conditions.propulsion.propeller_power                 = P*num_engines
        conditions.propulsion.propeller_thrust_coefficient    = Cp   
        conditions.propulsion.propeller_efficiency            = etap       
        conditions.propulsion.propeller_thrust_coefficient    = outputs.thrust_coefficient  
        
        
        # Compute force vector       
        F_vec = self.number_of_engines * F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]   
        
        F_mag = np.atleast_2d(np.linalg.norm(F_vec, axis=1)) 
  
        conditions.propulsion.disc_loading                    = (F_mag.T)/(num_engines*np.pi*(R)**2) # N/m^2  
        conditions.propulsion.power_loading                   = (F_mag.T)/(P)    # N/W         
        
        mdot = state.ones_row(1)*0.0

        results = Data()
        results.thrust_force_vector = F_vec
        results.vehicle_mass_rate   = mdot   
        
        return results
      
    def unpack_unknowns(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.propeller_power_coefficient [None]
            state.unknowns.battery_voltage_under_load  [volts]
    
            Outputs:
            state.conditions.propulsion.propeller_power_coefficient [None]
            state.conditions.propulsion.battery_voltage_under_load  [volts]
    
            Properties Used:
            N/A
        """                  

        # Here we are going to unpack the unknowns (Cp) provided for this network
        segment.state.conditions.propulsion.propeller_power_coefficient = segment.state.unknowns.propeller_power_coefficient
        segment.state.conditions.propulsion.battery_voltage_under_load  = segment.state.unknowns.battery_voltage_under_load
        segment.state.conditions.propulsion.throttle                    = segment.state.unknowns.throttle
          
        
        return
    
    def residuals(self,segment):
        """ This packs the residuals to be send to the mission solver.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                motor_torque                          [N-m]
                propeller_torque                      [N-m]
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
            
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """        
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        
        # Unpack
        q_motor   = segment.state.conditions.propulsion.motor_torque
        q_prop    = segment.state.conditions.propulsion.propeller_torque
        v_actual  = segment.state.conditions.propulsion.voltage_under_load
        v_predict = segment.state.unknowns.battery_voltage_under_load
        v_max     = self.voltage
        
        # Return the residuals
        segment.state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0]
        segment.state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max 
        
        return    
            
    __call__ = evaluate_thrust

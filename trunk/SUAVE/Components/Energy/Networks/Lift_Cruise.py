## @ingroup Components-Energy-Networks
# Lift_Cruise.py
# 
# Created: Jan 2016, E. Botero
#          Mar 2020, M. Clarke
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
from SUAVE.Core import Units, Data
from SUAVE.Components.Propulsors.Propulsor import Propulsor

# ----------------------------------------------------------------------
#  Lift_Forward
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Lift_Cruise(Propulsor):
    """ This is a complex version of battery_propeller with a battery powering propellers through
        electric motors. In this case we have 2 sets of motors at different motors that can be controlled seperately
        
        This network adds 2 extra unknowns to the mission. The first is
        a voltage, to calculate the thevenin voltage drop in the pack.
        The second is torque matching between motor and propeller.
        
        We have two inputs, the forward propeller throttle and the lift rotor throttle setting
        Since this is an airplane first and foremost, the "throttle" will be for forward thrust
        The new unknown will be for rotor throttle, because of the assumption on throttle something needs to be done...
        Want only 1 residual on voltage
    
        Assumptions:
        For any segment using this, body angle can't be an unknown.
        
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
        self.rotor_motor                 = None
        self.propeller_motor             = None
        self.rotor                       = None
        self.propeller                   = None
        self.esc_lift                    = None
        self.esc_forward                 = None
        self.avionics                    = None
        self.payload                     = None
        self.battery                     = None
        self.rotor_nacelle_diameter      = None
        self.propeller_diameter          = None
        self.rotor_engine_length         = None
        self.propeller_engine_length     = None
        self.propeller_engine_length     = None
        self.number_of_rotor_engines     = None
        self.number_of_propeller_engines = None
        self.voltage                     = None
        self.rotor_thrust_angle          = 0.0
        self.propeller_thrust_angle      = 0.0
        self.tag                         = 'Lift_Cruise'
        self.generative_design_minimum   = 0        
        pass
        
    def evaluate_thrust(self,state):
        """ Calculate thrust given the current state of the vehicle
    
            Assumptions:
            Caps the throttle at 110% and linearly interpolates thrust off that
    
            Source:
            N/A
    
            Inputs:
            state [state()]
    
            Outputs:
            results.thrust_force_vector [Newtons]
            results.vehicle_mass_rate   [kg/s]
            conditions.propulsion:
                rotor_rpm                 [radians/sec]
                propeller_rpm             [radians/sec]
                rotor_current             [amps]
                propeller_current          [amps]
                battery_draw             [watts]
                battery_energy           [joules]
                voltage_open_circuit     [volts]
                voltage_under_load       [volts]
                rotor_torque        [N-M]
                propeller_torque     [N-M]
                rotor_torque    [N-M]
                propeller_torque_forward [N-M]
    
            Properties Used:
            Defaulted values
        """          
        
        # unpack
        conditions        = state.conditions
        numerics          = state.numerics
        rotor_motor       = self.rotor_motor 
        propeller_motor   = self.propeller_motor
        rotor             = self.rotor 
        propeller         = self.propeller
        esc_lift          = self.esc_lift
        esc_forward       = self.esc_forward        
        avionics          = self.avionics
        payload           = self.payload
        battery           = self.battery
        num_lift          = self.number_of_rotor_engines
        num_forward       = self.number_of_propeller_engines
        
        #-----------------------------------------------------------------
        # SETUP BATTERIES AND ESC's
        #-----------------------------------------------------------------
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy    
        
        volts = state.unknowns.battery_voltage_under_load * 1. 
        volts[volts>self.voltage] = self.voltage 
        
        # ESC Voltage
        esc_lift.inputs.voltagein    = volts      
        esc_forward.inputs.voltagein = volts 
        
        #---------------------------------------------------------------
        # EVALUATE THRUST FROM FORWARD PROPULSORS 
        #---------------------------------------------------------------
        # Throttle the voltage
        esc_forward.voltageout(conditions) 
        
        # link
        propeller_motor.inputs.voltage = esc_forward.outputs.voltageout
        
        # Run the motor
        propeller_motor.omega(conditions)
        
        # link
        propeller.inputs.omega = propeller_motor.outputs.omega
        propeller.thrust_angle = self.propeller_thrust_angle   
        
        # Run the propeller
        F_forward, Q_forward, P_forward, Cp_forward, noise_forward, etap_forward = propeller.spin(conditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = conditions.propulsion.throttle[:,0,None]
        P_forward[eta>1.0] = P_forward[eta>1.0]*eta[eta>1.0]
        F_forward[eta>1.0] = F_forward[eta>1.0]*eta[eta>1.0]        
        
        # Run the motor for current
        i, etam_forward = propeller_motor.current(conditions)  
        
        # Fix the current for the throttle cap
        propeller_motor.outputs.current[eta>1.0] = propeller_motor.outputs.current[eta>1.0]*eta[eta>1.0]
        
        # link
        esc_forward.inputs.currentout =  propeller_motor.outputs.current 
        
        # Run the esc
        esc_forward.currentin(conditions)        
       
        #-------------------------------------------------------------------
        # EVALUATE THRUST FROM LIFT PROPULSORS 
        #-------------------------------------------------------------------
        
        # Make a new set of konditions, since there are differences for the esc and motor
        konditions                                        = Data()
        konditions.propulsion                             = Data()
        konditions.freestream                             = Data()
        konditions.frames                                 = Data()
        konditions.frames.inertial                        = Data()
        konditions.frames.body                            = Data()
        konditions.propulsion.acoustic_outputs            = Data()                
        konditions.propulsion.throttle                    = conditions.propulsion.rotor_throttle* 1.
        konditions.propulsion.propeller_power_coefficient = conditions.propulsion.rotor_power_coefficient * 1.
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
        rotor_motor.inputs.voltage = esc_lift.outputs.voltageout
        
        # Run the motor
        rotor_motor.omega(konditions)
        # link
        rotor.inputs.omega = rotor_motor.outputs.omega
        rotor.thrust_angle = self.rotor_thrust_angle
        
        # Run the propeller
        F_lift, Q_lift, P_lift, Cp_lift, output_lift, etap_lift = rotor.spin(konditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta = state.conditions.propulsion.rotor_throttle
        P_lift[eta>1.0] = P_lift[eta>1.0]*eta[eta>1.0]
        F_lift[eta>1.0] = F_lift[eta>1.0]*eta[eta>1.0]        
        
        # Run the motor for current
        i, etam_lift = rotor_motor.current(konditions)  
        
        # Fix the current for the throttle cap
        rotor_motor.outputs.current[eta>1.0] = rotor_motor.outputs.current[eta>1.0]*eta[eta>1.0]
        
        # link
        esc_lift.inputs.currentout =  rotor_motor.outputs.current     
        
        # Run the esc
        esc_lift.currentin(konditions)          
        
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
        battery.energy_calc(numerics)   
        
        # Pack the conditions
        a                    = conditions.freestream.speed_of_sound
        R_lift               = rotor.tip_radius
        R_forward            = propeller.tip_radius
        rotor_rpm            = rotor_motor.outputs.omega / Units.rpm
        propeller_rpm        = propeller_motor.outputs.omega / Units.rpm 
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy 
        voltage_open_circuit = battery.voltage_open_circuit
        voltage_under_load   = battery.voltage_under_load    
        
        # pack results for acoustic analysis
        conditions.propulsion.acoustic_outputs[propeller.tag]   = noise_forward
        conditions.propulsion.acoustic_outputs[rotor.tag]       = output_lift 
        
        # pack results of rotor 
        conditions.propulsion.rotor_rpm                         = rotor_rpm
        conditions.propulsion.rotor_current                     = i_lift 
        conditions.propulsion.rotor_motor_torque                = rotor_motor.outputs.torque
        conditions.propulsion.rotor_motor_efficiency            = etam_lift           
        conditions.propulsion.rotor_torque                      = Q_lift 
        conditions.propulsion.rotor_tip_mach                    = (rotor_motor.outputs.omega * R_lift )/a
        conditions.propulsion.rotor_efficiency                  = etap_lift
        conditions.propulsion.rotor_power                       = P_lift*num_lift
        conditions.propulsion.rotor_thrust                      = F_lift*num_lift        
        conditions.propulsion.rotor_power_coefficient           = Cp_lift    
        conditions.propulsion.rotor_thrust_coefficien           = output_lift.thrust_coefficient  
        conditions.propulsion.rotor_battery_draw                = -i_lift * volts  
          
        # pack results of propeller                                                     
        conditions.propulsion.propeller_motor_torque            = propeller_motor.outputs.torque  
        conditions.propulsion.propeller_rpm                     = propeller_rpm   
        conditions.propulsion.propeller_current                 = i_forward
        conditions.propulsion.propeller_battery_draw            = -i_forward * volts 
        conditions.propulsion.propeller_motor_efficiency        = etam_forward
        conditions.propulsion.propeller_current                 = i_forward 
        conditions.propulsion.propeller_power                   = P_forward*num_forward
        conditions.propulsion.propeller_thrust                  = F_forward*num_forward
        conditions.propulsion.propeller_thrust_coefficient      = Cp_forward 
        conditions.propulsion.propeller_tip_mach                = (propeller_motor.outputs.omega * R_forward)/a
        conditions.propulsion.propeller_torque                  = Q_forward       
        conditions.propulsion.propeller_efficiency              = etap_forward 

        conditions.propulsion.battery_energy                    = battery_energy
        conditions.propulsion.battery_efficiency                = (battery_draw+battery.resistive_losses)/battery_draw
        conditions.propulsion.voltage_open_circuit              = voltage_open_circuit
        conditions.propulsion.voltage_under_load                = voltage_under_load           
        conditions.propulsion.payload_efficiency                = (battery_draw+(avionics.outputs.power + payload.outputs.power))/battery_draw            
        conditions.propulsion.current                           = i_lift + i_forward 
        conditions.propulsion.battery_specfic_power             = -battery_draw/battery.mass_properties.mass    # kWh/kg
        conditions.propulsion.battery_current                   = current_total 
        conditions.propulsion.electronics_efficiency            = -(P_forward*num_forward+P_lift*num_lift)/battery_draw  
        conditions.propulsion.battery_draw                      = battery_draw
        
        # Calculate the thrust and mdot
        F_lift_total    = F_lift*num_lift * [np.cos(self.rotor_thrust_angle),0,-np.sin(self.rotor_thrust_angle)]    
        F_forward_total = F_forward*num_forward * [np.cos(self.propeller_thrust_angle),0,-np.sin(self.propeller_thrust_angle)] 


        F_lift_mag    = np.atleast_2d(np.linalg.norm(F_lift_total, axis=1))
        F_forward_mag = np.atleast_2d(np.linalg.norm(F_forward_total, axis=1))
        
        conditions.propulsion.disc_loading_lift                 = (F_lift_mag.T)/(self.number_of_rotor_engines*np.pi*(R_lift)**2) # N/m^2              
        conditions.propulsion.disc_loading_forward              = (F_forward_mag.T)/(self.number_of_propeller_engines*np.pi*(R_forward)**2)  # N/m^2      
        conditions.propulsion.power_loading_lift                = (F_lift_mag.T)/(battery_draw)      # N/W 
        conditions.propulsion.power_loading_forward             = (F_forward_mag.T)/(battery_draw)   # N/W    
                                                                                                        
        F_total = F_lift_total + F_forward_total
        mdot = state.ones_row(1)*0.0
        
        results = Data()
        results.thrust_force_vector = F_total
        results.vehicle_mass_rate   = mdot    
        
        return results
    
    def unpack_unknowns_transition(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
            This uses all the motors.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.rotor_power_coefficient[None]
            state.unknowns.propeller_power_coefficient [None]
            state.unknowns.battery_voltage_under_load  [volts]
            state.unknowns.rotor_throttle               [0-1]
            state.unknowns.throttle                    [0-1]
    
            Outputs:
            state.conditions.propulsion.rotor_power_coefficient [None]
            state.conditions.propulsion.propeller_power_coefficient [None]
            state.conditions.propulsion.battery_voltage_under_load  [volts]
            state.conditions.propulsion.rotor_throttle              [0-1]
            state.conditions.propulsion.throttle                    [0-1]
    
            Properties Used:
            N/A
        """          
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        segment.state.conditions.propulsion.battery_voltage_under_load        = segment.state.unknowns.battery_voltage_under_load
        segment.state.conditions.propulsion.rotor_power_coefficient           = segment.state.unknowns.rotor_power_coefficient
        segment.state.conditions.propulsion.propeller_power_coefficient       = segment.state.unknowns.propeller_power_coefficient   
        segment.state.conditions.propulsion.rotor_throttle                    = segment.state.unknowns.rotor_throttle        
        segment.state.conditions.propulsion.throttle                          = segment.state.unknowns.throttle
        
        return
    
    
    def unpack_unknowns_no_lift(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
            This uses only the forward motors and turns the rest off.
    
            Assumptions:
            Only the forward motors and turns the rest off.
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.propeller_power_coefficient [None]
            state.unknowns.battery_voltage_under_load  [volts]
            state.unknowns.rotor_throttle              [0-1]
            state.unknowns.throttle                    [0-1]
    
            Outputs:
            state.conditions.propulsion.propeller_power_coefficient [None]
            state.conditions.propulsion.battery_voltage_under_load  [volts]
            state.conditions.propulsion.rotor_throttle              [0-1]
            state.conditions.propulsion.throttle                    [0-1]
    
            Properties Used:
            N/A
        """             
        
        ones = segment.state.ones_row
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        segment.state.conditions.propulsion.rotor_throttle                      = 0.0 * ones(1)
        segment.state.conditions.propulsion.battery_voltage_under_load          = segment.state.unknowns.battery_voltage_under_load
        segment.state.conditions.propulsion.propeller_power_coefficient         = segment.state.unknowns.propeller_power_coefficient
        segment.state.conditions.propulsion.rotor_power_coefficient             = 0.0 * ones(1)
        segment.state.conditions.propulsion.throttle                            = segment.state.unknowns.throttle
        
        return    
    
    def unpack_unknowns_no_forward(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
            This uses only the lift motors.
    
            Assumptions:
            Only the lift motors.
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.propeller_power_coefficient [None]
            state.unknowns.battery_voltage_under_load  [volts]
            state.unknowns.rotor_throttle              [0-1]
            state.unknowns.throttle            [0-1]
    
            Outputs:
            state.conditions.propulsion.propeller_power_coefficient [None]
            state.conditions.propulsion.battery_voltage_under_load  [volts]
            state.conditions.propulsion.rotor_throttle              [0-1]
            state.conditions.propulsion.throttle            [0-1]
    
            Properties Used:
            N/A
        """             
        
        ones = segment.state.ones_row
        
        # Here we are going to unpack the unknowns (Cps,throttle,voltage) provided for this network
        segment.state.conditions.propulsion.rotor_throttle                       = segment.state.unknowns.rotor_throttle
        segment.state.conditions.propulsion.battery_voltage_under_load           = segment.state.unknowns.battery_voltage_under_load
        segment.state.conditions.propulsion.propeller_power_coefficient          = 0.0 * ones(1)
        segment.state.conditions.propulsion.rotor_power_coefficient              = segment.state.unknowns.rotor_power_coefficient
        segment.state.conditions.propulsion.throttle                             = 0.0 * ones(1)
        
        return    
    
    def residuals_transition(self,segment):
        """ This packs the residuals to be send to the mission solver.
            Use this if all motors are operational
    
            Assumptions:
            All motors are operational
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                propeller_torque                      [N-m]
                rotor_torque                          [N-m]
                propeller_torque_forward              [N-m]
                rotor_torque                          [N-m]
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
    
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """            
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        q_propeller_motor = segment.state.conditions.propulsion.propeller_motor_torque
        q_prop_forward    = segment.state.conditions.propulsion.propeller_torque 
        q_rotor_motor     = segment.state.conditions.propulsion.rotor_motor_torque
        q_prop_lift       = segment.state.conditions.propulsion.rotor_torque        
        
        v_actual          = segment.state.conditions.propulsion.voltage_under_load
        v_predict         = segment.state.unknowns.battery_voltage_under_load
        v_max             = self.voltage        
        
        # Return the residuals
        segment.state.residuals.network[:,0] = (q_propeller_motor[:,0] - q_prop_forward[:,0])/q_propeller_motor[:,0] 
        segment.state.residuals.network[:,1] = (q_rotor_motor[:,0] - q_prop_lift[:,0])/q_rotor_motor[:,0]
        segment.state.residuals.network[:,2] = (v_predict[:,0] - v_actual[:,0])/v_max  

        return
    
    
    def residuals_no_lift(self,segment):
        """ This packs the residuals to be send to the mission solver.
            Use this if only the forward motors are operational
    
            Assumptions:
            Only the forward motors are operational
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                propeller_torque                      [N-m]
                rotor_torque                          [N-m]
                propeller_torque_forward              [N-m]
                rotor_torque                          [N-m]
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
            
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """          
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        q_propeller_motor = segment.state.conditions.propulsion.propeller_motor_torque
        q_prop_forward    = segment.state.conditions.propulsion.propeller_torque  
        
        v_actual          = segment.state.conditions.propulsion.voltage_under_load
        v_predict         = segment.state.unknowns.battery_voltage_under_load
        v_max             = self.voltage        
        
        # Return the residuals
        segment.state.residuals.network[:,0] = (q_propeller_motor[:,0] - q_prop_forward[:,0])/q_propeller_motor[:,0]
        segment.state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max 
        return    
    
    def residuals_no_forward(self,segment):
        """ This packs the residuals to be send to the mission solver.
            Only the lift motors are operational
    
            Assumptions:
            The lift motors are operational
    
            Source:
            N/A
    
            Inputs:
            state.conditions.propulsion:
                propeller_torque                      [N-m]
                rotor_torque                          [N-m]
                propeller_torque_forward              [N-m]
                rotor_torque                          [N-m]
                voltage_under_load                    [volts]
            state.unknowns.battery_voltage_under_load [volts]
    
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """            
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        q_rotor_motor   = segment.state.conditions.propulsion.rotor_motor_torque
        q_prop_lift     = segment.state.conditions.propulsion.rotor_torque        

        v_actual        = segment.state.conditions.propulsion.voltage_under_load
        v_predict       = segment.state.unknowns.battery_voltage_under_load
        v_max           = self.voltage        
        
        # Return the residuals
        segment.state.residuals.network[:,0] = (q_rotor_motor[:,0] - q_prop_lift[:,0])/q_rotor_motor[:,0]
        segment.state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max  
        return
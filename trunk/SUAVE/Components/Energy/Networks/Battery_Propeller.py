## @ingroup Components-Energy-Networks
# Battery_Propeller.py
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
from SUAVE.Methods.Power.Battery.Discharge.thevenin_discharge import battery_performance_maps

from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Network
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Networks
class Battery_Propeller(Propulsor):
    """ This is a simple network with a battery powering a propeller through
        an electric motor
        
        This network adds 2 extra unknowns to the mission. The first is
        a voltage, to calculate the thevenin voltage drop in the pack.
        The second is torque matching between motor and propeller.
    
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
        self.thrust_angle      = 0.0
        self.dischage_model_fidelity = 1
        self.use_surrogate     = False
    
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
        conditions         = state.conditions
        numerics           = state.numerics
        motor              = self.motor
        propeller          = self.propeller
        esc                = self.esc
        avionics           = self.avionics
        payload            = self.payload
        battery            = self.battery
        D                  = numerics.time.differentiate    
        I                  = numerics.time.integrate        
        dischage_fidelity  = self.dischage_model_fidelity 
        battery_data       = battery_performance_maps() 
        num_engines        = self.number_of_engines
        
        # Set battery energy
        battery.current_energy = conditions.propulsion.battery_energy  

        if dischage_fidelity == 1: 
            volts = state.unknowns.battery_voltage_under_load
            battery.battery_thevenin_voltage = 0 
        elif dischage_fidelity == 2: 
            # calculate the current going into one cell 
            n_series   = battery.module_config[0]  
            n_parallel = battery.module_config[1]
            n_total    = n_series * n_parallel  
  
            Tbat = battery.temperature
            SOC  = state.unknowns.battery_state_of_charge 
            V_Th = state.unknowns.battery_thevenin_voltage/n_series 
            
            # look up tables 
            V_oc = np.zeros_like(SOC)
            R_Th = np.zeros_like(SOC)  
            C_Th = np.zeros_like(SOC)  
            R_0  = np.zeros_like(SOC)
            SOC[SOC<0] = 0
            for i in range(len(SOC)): 
                V_oc[i] = battery_data.V_oc_interp(Tbat, SOC[i])[0]
                C_Th[i] = battery_data.C_Th_interp(Tbat, SOC[i])[0]
                R_Th[i] = battery_data.R_Th_interp(Tbat, SOC[i])[0]
                R_0[i]  = battery_data.R_0_interp(Tbat, SOC[i])[0]
            
            dV_TH_dt =  np.dot(D,V_Th)
            Icell = V_Th/R_Th  + C_Th*dV_TH_dt
            
            # Voltage under load:
            volts =  n_series*(V_oc - V_Th - (Icell * R_0))
            
        # Step 1 battery power
        esc.inputs.voltagein = volts
        
        # Step 2
        esc.voltageout(conditions)
        
        # link
        motor.inputs.voltage = esc.outputs.voltageout 
        
        # step 3
        motor.omega(conditions)
        
        # link
        propeller.inputs.omega              = motor.outputs.omega
        propeller.thrust_angle              = self.thrust_angle
        conditions.propulsion.pitch_command = self.pitch_command
        
        # step 4
        F, Q, P, Cp, outputs , etap = propeller.spin_variable_pitch(conditions)
            
        # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
        eta        = conditions.propulsion.throttle[:,0,None]
        P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
        F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
        
        # link
        propeller.outputs = outputs
        
        # Run the avionics
        avionics.power()

        # Run the payload
        payload.power()

        # Run the motor for current
        motor.current(conditions)
        
        # link
        esc.inputs.currentout =  motor.outputs.current

        # Run the esc
        esc.currentin(conditions)

        # Calculate avionics and payload power
        avionics_payload_power = avionics.outputs.power + payload.outputs.power

        # Calculate avionics and payload current
        avionics_payload_current = avionics_payload_power/self.voltage

        # link
        battery.inputs.current  = esc.outputs.currentin*self.number_of_engines + avionics_payload_current
        battery.inputs.power_in = -(esc.outputs.voltageout*esc.outputs.currentin*self.number_of_engines + avionics_payload_power)
        battery.energy_calc(numerics,fidelity = dischage_fidelity) 
    
        # Pack the conditions for outputs
        a                         = conditions.freestream.speed_of_sound
        R                         = propeller.tip_radius
        rpm                       = motor.outputs.omega*60./(2.*np.pi)
        current                   = esc.outputs.currentin
        battery_draw              = battery.inputs.power_in
        state_of_charge           = battery.state_of_charge
        battery_energy            = battery.current_energy
        voltage_open_circuit      = battery.voltage_open_circuit
        voltage_under_load        = battery.voltage_under_load 
        battery_thevenin_voltage  = battery.battery_thevenin_voltage
          
        conditions.propulsion.rpm                      = rpm
        conditions.propulsion.current                  = current
        conditions.propulsion.battery_draw             = battery_draw
        conditions.propulsion.battery_energy           = battery_energy
        conditions.propulsion.voltage_open_circuit     = voltage_open_circuit
        conditions.propulsion.voltage_under_load       = voltage_under_load  
        conditions.propulsion.state_of_charge          = state_of_charge
        conditions.propulsion.motor_torque             = motor.outputs.torque
        conditions.propulsion.motor_temperature        = 0 # currently no motor temperature model
        conditions.propulsion.propeller_torque         = Q
        conditions.propulsion.battery_thevenin_voltage = battery_thevenin_voltage
        conditions.propulsion.battery_specfic_power    = -(battery_draw/1000)/battery.mass_properties.mass 
        conditions.propulsion.propeller_tip_mach       = (R*rpm)/a
        
        # Create the outputs
        F    = self.number_of_engines * F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]      
        mdot = np.zeros_like(F)

        F_mag = np.atleast_2d(np.linalg.norm(F, axis=1)*0.224809) # lb   
        conditions.propulsion.disc_loading          = (F_mag.T)/ (num_engines*np.pi*(R*3.28084)**2) # lb/ft^2                     
        conditions.propulsion.power_loading         = (F_mag.T)/(battery_draw*0.00134102)           # lb/hp 
        
        results = Data()
        results.thrust_force_vector = F
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
        
        
    
    def unpack_unknowns_thevenin(self,segment): 
        
        segment.state.conditions.propulsion.propeller_power_coefficient = segment.state.unknowns.propeller_power_coefficient
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_thevenin_voltage = segment.state.unknowns.battery_thevenin_voltage 
        
        return
    
    def residuals_thevenin(self,segment):  
        
        # Unpack
        q_motor   = segment.state.conditions.propulsion.motor_torque
        q_prop    = segment.state.conditions.propulsion.propeller_torque 
        SOC_actual  = segment.state.conditions.propulsion.state_of_charge
        SOC_predict = segment.state.unknowns.battery_state_of_charge  
        v_th_actual  = segment.state.conditions.propulsion.battery_thevenin_voltage
        v_th_predict = segment.state.unknowns.battery_thevenin_voltage 
        
        # Return the residuals 
        segment.state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0] 
        segment.state.residuals.network[:,1] = v_th_predict[:,0] - v_th_actual[:,0]     
        segment.state.residuals.network[:,2] = SOC_predict[:,0] - SOC_actual[:,0]  
        
        
    __call__ = evaluate_thrust



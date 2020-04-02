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
        self.motor                   = None
        self.propeller               = None
        self.esc                     = None
        self.avionics                = None
        self.payload                 = None
        self.battery                 = None
        self.nacelle_diameter        = None
        self.engine_length           = None
        self.number_of_engines       = None
        self.voltage                 = None
        self.thrust_angle            = 0.0
        self.pitch_command           = 0.0 
        self.use_surrogate           = False
    
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
                battery_voltage_open_circuit [volts]
                battery_voltage_under_load    [volts]
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
        battery_data       = battery.discharge_performance_map
        num_engines        = self.number_of_engines
         
        # Set battery energy
        battery.current_energy      = conditions.propulsion.battery_energy
        battery.temperature         = conditions.propulsion.battery_temperature
        battery.charge_throughput   = conditions.propulsion.battery_charge_throughput
        battery.ambient_temperature = conditions.propulsion.ambient_temperature          
        battery.age_in_days         = conditions.propulsion.battery_age_in_days 
        discharge_flag              = conditions.propulsion.battery_discharge    
        battery.R_growth_factor     = conditions.propulsion.battery_resistance_growth_factor
        battery.E_growth_factor     = conditions.propulsion.battery_capacity_fade_factor 
        battery.max_energy          = battery.initial_max_energy * battery.E_growth_factor       

        
        if battery.chemistry == 'LiNCA':  
            n_series                    = battery.module_config[0]  
            n_parallel                  = battery.module_config[1]
            n_total                     = n_series * n_parallel  
            
            SOC    = state.unknowns.battery_state_of_charge 
            T_cell = state.unknowns.battery_cell_temperature 
            V_Th   = state.unknowns.battery_thevenin_voltage/n_series 
            
            # link temperature 
            battery.cell_temperature = T_cell     
            
            # look up tables  
            V_oc = np.zeros_like(SOC)
            R_Th = np.zeros_like(SOC)  
            C_Th = np.zeros_like(SOC)  
            R_0  = np.zeros_like(SOC)
            SOC[SOC<0.] = 0.
            SOC[SOC>1.] = 1.
            for i in range(len(SOC)): 
                V_oc[i] = battery_data.V_oc_interp(T_cell[i], SOC[i])[0]
                C_Th[i] = battery_data.C_Th_interp(T_cell[i], SOC[i])[0]
                R_Th[i] = battery_data.R_Th_interp(T_cell[i], SOC[i])[0]
                R_0[i]  = battery_data.R_0_interp(T_cell[i], SOC[i])[0]  
                
            dV_TH_dt =  np.dot(D,V_Th)
            Icell = V_Th/(R_Th * battery.R_growth_factor)  + C_Th*dV_TH_dt 
            I_tot = Icell * n_parallel 
            R_0   = R_0 * battery.R_growth_factor 
             
            # Voltage under load:
            volts =  n_series*(V_oc - V_Th - (Icell * R_0)) 


        elif battery.chemistry == 'LiNiMnCoO2':  
            volts               = state.unknowns.battery_voltage_under_load 
            battery.temperature =  state.unknowns.battery_cell_temperature    
 
        else: 
            volts                            = state.unknowns.battery_voltage_under_load
            battery.battery_thevenin_voltage = 0  
            battery.temperature              = conditions.propulsion.battery_temperature             
            battery.cell_temperature         = battery.temperature 

        if discharge_flag:     
            # Run the avionics
            avionics.power()
        
            # Run the payload
            payload.power()
            
            # Calculate avionics and payload power
            avionics_payload_power = avionics.outputs.power + payload.outputs.power
        
            # Calculate avionics and payload current
            avionics_payload_current = avionics_payload_power/volts
                    
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
            F, Q, P, Cp, outputs , etap = propeller.spin(conditions)
                
            # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
            eta        = conditions.propulsion.throttle[:,0,None]
            P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
            F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
            
            # link
            propeller.outputs = outputs
            
            # Run the motor for current
            motor.current(conditions)
            
            # link
            esc.inputs.currentout =   motor.outputs.current
    
            # Run the esc
            esc.currentin(conditions) 
    
            # link
            battery.inputs.current  = esc.outputs.currentin*self.number_of_engines + avionics_payload_current
            battery.inputs.power_in = -(volts *esc.outputs.currentin*self.number_of_engines + avionics_payload_power)
            battery.inputs.voltage  = volts
            battery.energy_discharge(numerics)          
            
        else:  
            # link 
            battery.inputs.current  = -(battery.charging_current * n_parallel) * np.ones_like(volts)
            battery.inputs.power_in =  (battery.charging_current * n_parallel) * volts * np.ones_like(volts)
            battery.inputs.voltage  = volts 
            Q = np.zeros_like(volts)
            F = np.zeros_like(volts)
            battery.energy_charge(numerics)        
            conditions.propulsion.propeller_power_coefficient  = np.zeros_like(volts)
            conditions.propulsion.etap  = np.zeros_like(volts)
            conditions.propulsion.etam  = np.zeros_like(volts)
            
        # Pack the conditions for outputs
        a                         = conditions.freestream.speed_of_sound
        R                         = propeller.tip_radius
        rpm                       = motor.outputs.omega*60./(2.*np.pi)
        battery_draw              = battery.inputs.power_in  
        
        conditions.propulsion.rpm                          = rpm
        conditions.propulsion.battery_current              = battery.inputs.current
        conditions.propulsion.battery_draw                 = battery_draw
        conditions.propulsion.battery_energy               = battery.current_energy
        conditions.propulsion.battery_voltage_open_circuit = battery.voltage_open_circuit 
        conditions.propulsion.battery_voltage_under_load   = battery.voltage_under_load  
        conditions.propulsion.battery_charge_throughput    = battery.charge_throughput   
        conditions.propulsion.battery_state_of_charge      = battery.state_of_charge
        conditions.propulsion.battery_cell_temperature     = battery.cell_temperature
        conditions.propulsion.battery_specfic_power        = -battery_draw/battery.mass_properties.mass # Wh/kg
        conditions.propulsion.motor_torque                 = motor.outputs.torque
        conditions.propulsion.motor_temperature            = 0 # currently no motor temperature model
        conditions.propulsion.propeller_torque             = Q
        conditions.propulsion.battery_age_in_days          = battery.age_in_days 
        # Pack the conditions for outputs    
        if battery.chemistry == 'LiNCA':   
            conditions.propulsion.battery_thevenin_voltage               = battery.thevenin_voltage   
        conditions.propulsion.propeller_tip_mach        = (R*motor.outputs.omega)/a
        
        # Create the outputs  
        F                                           = num_engines* F * [np.cos(self.thrust_angle),0,-np.sin(self.thrust_angle)]      
        mdot                                        = np.zeros_like(F) 
        F_mag                                       = np.atleast_2d(np.linalg.norm(F, axis=1))  
        conditions.propulsion.disc_loading          = (F_mag.T)/ (num_engines*np.pi*(R)**2) # N/m^2                  
        conditions.propulsion.power_loading         = (F_mag.T)/(P)                         # N/W 
        
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
                battery_voltage_under_load            [volts]
            state.unknowns.battery_voltage_under_load [volts]
            
            Outputs:
            None
    
            Properties Used:
            self.voltage                              [volts]
        """        
        
        # Here we are going to pack the residuals (torque,voltage) from the network
        
        # Unpack
        q_motor      = segment.state.conditions.propulsion.motor_torque
        q_prop       = segment.state.conditions.propulsion.propeller_torque
        v_actual     = segment.state.conditions.propulsion.battery_voltage_under_load 
        v_predict    = segment.state.unknowns.battery_voltage_under_load               
        v_max        = self.voltage
        
        # Return the residuals
        segment.state.residuals.network[:,0] = (q_motor[:,0] - q_prop[:,0])/500
        segment.state.residuals.network[:,1] = (v_predict[:,0] - v_actual[:,0])/v_max      
        
        return    
    
    def unpack_unknowns_thevenin(self,segment):  
        segment.state.conditions.propulsion.propeller_power_coefficient = segment.state.unknowns.propeller_power_coefficient
        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_thevenin_voltage = segment.state.unknowns.battery_thevenin_voltage  
  
        return
    
    def residuals_thevenin(self,segment):  
        
        # Unpack
        q_motor      = segment.state.conditions.propulsion.motor_torque
        q_prop       = segment.state.conditions.propulsion.propeller_torque 
        
        SOC_actual  = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict = segment.state.unknowns.battery_state_of_charge 
        
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
        v_th_actual  = segment.state.conditions.propulsion.battery_thevenin_voltage
        v_th_predict = segment.state.unknowns.battery_thevenin_voltage        
       
        # Return the residuals 
        segment.state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0] 
        segment.state.residuals.network[:,1] = v_th_predict[:,0] - v_th_actual[:,0]     
        segment.state.residuals.network[:,2] = SOC_predict[:,0] - SOC_actual[:,0]  
        segment.state.residuals.network[:,3] = Temp_predict[:,0] - Temp_actual[:,0]
        
    def unpack_unknowns_linmco(self,segment): 
  
        segment.state.conditions.propulsion.propeller_power_coefficient = segment.state.unknowns.propeller_power_coefficient        
        segment.state.conditions.propulsion.battery_cell_temperature    = segment.state.unknowns.battery_cell_temperature
        segment.state.conditions.propulsion.battery_voltage_under_load  = segment.state.unknowns.battery_voltage_under_load
      
        return
    
    def residuals_linmco(self,segment):  
        # Unpack 
        q_motor      = segment.state.conditions.propulsion.motor_torque
        q_prop       = segment.state.conditions.propulsion.propeller_torque 
         
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
        v_actual     = segment.state.conditions.propulsion.battery_voltage_under_load 
        v_predict    = segment.state.unknowns.battery_voltage_under_load               
        v_max        = self.voltage
        
        # Return the residuals 
        segment.state.residuals.network[:,0] = (v_predict[:,0] - v_actual[:,0])/v_max      
        segment.state.residuals.network[:,1] = (Temp_predict[:,0] - Temp_actual[:,0])
        segment.state.residuals.network[:,2] =  (q_motor[:,0] - q_prop[:,0] ) 
        
        return           
    __call__ = evaluate_thrust



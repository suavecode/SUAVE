## @ingroup Components-Energy-Networks
# Vectored_Thrust.py
# 
# Created:  Nov 2018, M. Clarke
# Modified: Mar 2020, M. Clarke
#           May 2021, M. Clarke
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
        conditions    = state.conditions
        numerics      = state.numerics
        motor         = self.motor
        rotor         = self.rotor
        esc           = self.esc
        avionics      = self.avionics
        payload       = self.payload
        battery       = self.battery
        battery_data  = battery.discharge_performance_map
        num_engines   = self.number_of_engines
        D             = numerics.time.differentiate           
        t_nondim      = state.numerics.dimensionless.control_points
        
          # Set battery energy
        battery.current_energy      = conditions.propulsion.battery_energy
        battery.pack_temperature    = conditions.propulsion.battery_pack_temperature
        battery.charge_throughput   = conditions.propulsion.battery_cumulative_charge_throughput     
        battery.age_in_days         = conditions.propulsion.battery_age_in_days 
        discharge_flag              = conditions.propulsion.battery_discharge    
        battery.R_growth_factor     = conditions.propulsion.battery_resistance_growth_factor
        battery.E_growth_factor     = conditions.propulsion.battery_capacity_fade_factor 
        battery.max_energy          = conditions.propulsion.battery_max_aged_energy
        V_th0                       = conditions.propulsion.battery_initial_thevenin_voltage
        n_series                    = battery.pack_config.series  
        n_parallel                  = battery.pack_config.parallel
        n_total                     = n_series*n_parallel
        
        # update ambient temperature based on altitude
        battery.ambient_temperature                   = conditions.freestream.temperature   
        battery.cooling_fluid.thermal_conductivity    = conditions.freestream.thermal_conductivity
        battery.cooling_fluid.kinematic_viscosity     = conditions.freestream.kinematic_viscosity
        battery.cooling_fluid.density                 = conditions.freestream.density
        

        # --------------------------------------------------------------------------------
        # Predict Voltage and Battery Properties Depending on Battery Chemistry
        # -------------------------------------------------------------------------------- 
        if battery.chemistry == 'LiNCA':  
            
            SOC       = state.unknowns.battery_state_of_charge
            T_cell    = state.unknowns.battery_cell_temperature
            V_Th_cell = state.unknowns.battery_thevenin_voltage/n_series
            
            # link temperature 
            battery.cell_temperature = T_cell     
            
            # look up tables  
            V_oc_cell = np.zeros_like(SOC)
            R_Th_cell = np.zeros_like(SOC)
            C_Th_cell = np.zeros_like(SOC)
            R_0_cell  = np.zeros_like(SOC)
            SOC[SOC<0.] = 0.
            SOC[SOC>1.] = 1.
            for i in range(len(SOC)): 
                T_cell_Celcius = T_cell[i] - 272.65
                V_oc_cell[i] = battery_data.V_oc_interp(T_cell_Celcius, SOC[i])[0]
                C_Th_cell[i] = battery_data.C_Th_interp(T_cell_Celcius, SOC[i])[0]
                R_Th_cell[i] = battery_data.R_Th_interp(T_cell_Celcius, SOC[i])[0]
                R_0_cell[i]  = battery_data.R_0_interp( T_cell_Celcius, SOC[i])[0]  
                
            dV_TH_dt =  np.dot(D,V_Th_cell)
            I_cell   = V_Th_cell/(R_Th_cell * battery.R_growth_factor)  + C_Th_cell*dV_TH_dt
            R_0_cell = R_0_cell * battery.R_growth_factor
             
            # Voltage under load:
            volts =  n_series*(V_oc_cell - V_Th_cell - (I_cell  * R_0_cell)) 

        elif battery.chemistry == 'LiNiMnCoO2':

            SOC        = state.unknowns.battery_state_of_charge 
            T_cell     = state.unknowns.battery_cell_temperature
            I_cell     = state.unknowns.battery_current/n_parallel 
            
            # Link Temperature 
            battery.cell_temperature         = T_cell  
            battery.initial_thevenin_voltage = V_th0  
            
            # Make sure things do not break by limiting current, temperature and current 
            SOC[SOC < 0.]            = 0.  
            SOC[SOC > 1.]            = 1.    
            DOD                      = 1 - SOC 
            
            T_cell[np.isnan(T_cell)] = 302.65
            T_cell[T_cell<272.65]    = 272.65 # model does not fit for below 0  degrees
            T_cell[T_cell>322.65]    = 322.65 # model does not fit for above 50 degrees
             
            I_cell[I_cell<0.0]       = 0.0
            I_cell[I_cell>8.0]       = 8.0   
            
            # create vector of conditions for battery data sheet response surface for OCV
            T_cell_Celcius           = T_cell  - 272.65
            pts                      = np.hstack((np.hstack((I_cell, T_cell_Celcius)),DOD  )) # amps, temp, SOC   
            V_ul_cell                = np.atleast_2d(battery_data.Voltage(pts)[:,1]).T   
            volts                    = n_series*V_ul_cell   
 
        else: 
            volts                            = state.unknowns.battery_voltage_under_load
            battery.battery_thevenin_voltage = 0             
            battery.cell_temperature         = battery.temperature  
            
       
        # --------------------------------------------------------------------------------
        # Run Motor, Avionics and Systems (Discharge Model)
        # --------------------------------------------------------------------------------    
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
        
            # Define the thrust angle 
            thrust_angle = self.thrust_angle
        
            # link
            rotor.inputs.omega  = motor.outputs.omega
            rotor.thrust_angle  = thrust_angle
            rotor.pitch_command = self.pitch_command  
            rotor.VTOL_flag     = state.VTOL_flag    
        
            # Run the rotor     
            F, Q, P, Cp , outputs, etap = rotor.spin(conditions)
            
            
            # Check to see if magic thrust is needed, the ESC caps throttle at 1.1 already
            eta        = conditions.propulsion.throttle[:,0,None]
            P[eta>1.0] = P[eta>1.0]*eta[eta>1.0]
            F[eta>1.0] = F[eta>1.0]*eta[eta>1.0]
        
            # link
            rotor.outputs = outputs 
            
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
            battery.inputs.voltage  = volts
            battery.energy_discharge(numerics)  
              
        # --------------------------------------------------------------------------------
        # Run Charge Model 
        # --------------------------------------------------------------------------------               
        else:  
            # link 
            battery.inputs.current  = -battery.cell.charging_current * n_parallel * np.ones_like(volts)
            battery.inputs.power_in =  battery.cell.charging_current * n_parallel * volts * np.ones_like(volts)
            battery.inputs.voltage  =  battery.charging_voltage 
            Q                       = np.zeros_like(volts)
            F                       = np.zeros_like(volts)
            Cp                      = np.zeros_like(volts)
            etap                    = np.zeros_like(volts) 
            etam                    = np.zeros_like(volts)
            battery.energy_charge(numerics)        
        
        # Pack the conditions for outputs
        rpm                  = motor.outputs.omega / Units.rpm
        a                    = conditions.freestream.speed_of_sound
        R                    = rotor.tip_radius       
        current              = esc.outputs.currentin
        battery_draw         = battery.inputs.power_in 
        battery_energy       = battery.current_energy
        voltage_open_circuit = battery.voltage_open_circuit
        voltage_under_load   = battery.voltage_under_load    
        state_of_charge      = battery.state_of_charge
        
          
        conditions.propulsion.rpm                             = rpm
        conditions.propulsion.current                         = current
        conditions.propulsion.battery_draw                    = battery_draw
        conditions.propulsion.battery_energy                  = battery_energy 
        conditions.propulsion.voltage_open_circuit            = voltage_open_circuit
        conditions.propulsion.voltage_under_load              = voltage_under_load  
        conditions.propulsion.state_of_charge                 = state_of_charge        
        conditions.propulsion.motor_torque                    = motor.outputs.torque
        conditions.propulsion.propeller_torque                = Q
        conditions.propulsion.propeller_motor_efficiency      = etam
        conditions.propulsion.acoustic_outputs[rotor.tag]     = outputs
        conditions.propulsion.battery_specfic_power           = -battery_draw/battery.mass_properties.mass #Wh/kg
        conditions.propulsion.electronics_efficiency          = -(P*num_engines)/battery_draw   
        conditions.propulsion.propeller_tip_mach              = (R*rpm*Units.rpm)/a
        conditions.propulsion.battery_current                 = total_current
        conditions.propulsion.battery_efficiency              = (battery_draw+battery.resistive_losses)/battery_draw
        conditions.propulsion.payload_efficiency              = (battery_draw+(avionics.outputs.power + payload.outputs.power))/battery_draw            
        conditions.propulsion.propeller_power                 = P*num_engines 
        conditions.propulsion.propeller_efficiency            = etap       
        conditions.propulsion.propeller_thrust_coefficient    = outputs.thrust_coefficient  
        conditions.propulsion.propeller_power_coefficient     = Cp 
        
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
    
     
    # -----------------------------------------------------------------
    # Generic Li-Battery Cell Unknows and Residuals 
    # -----------------------------------------------------------------       
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
    
    def unpack_unknowns_charge(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.

            Assumptions:
            None

            Source:
            N/A

            Inputs:
            state.unknowns.battery_voltage_under_load  [volts]

            Outputs:
            state.conditions.propulsion.battery_voltage_under_load  [volts]

            Properties Used:
            N/A
        """

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
    
    def unpack_unknowns_linca(self,segment):
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.
            This uses only the lift motors.
    
            Assumptions:
            Only the lift motors.
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.propeller_power_coefficient              [None]  
            state.unknowns.battery_cell_temperature                 [K]
            state.unknowns.battery_state_of_charge                  [None]
            state.unknowns.battery_current                          [A]
                                                                    
            Outputs:
            state.conditions.propulsion.propeller_power_coefficient [None]  
            state.conditions.propulsion.battery_cell_temperature    [K]
            state.conditions.propulsion.battery_state_of_charge     [None]
            state.conditions.propulsion.battery_current             [A]
    
            Properties Used:
            N/A
        """         
        segment.state.conditions.propulsion.propeller_power_coefficient = segment.state.unknowns.propeller_power_coefficient
        segment.state.conditions.propulsion.battery_cell_temperature    = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge     = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_thevenin_voltage    = segment.state.unknowns.battery_thevenin_voltage  
  
        return
    
    def residuals_linca(self,segment):  
        """ This packs the residuals to be send to the mission solver.

            Assumptions:
            None

            Source:
            N/A

            Inputs: 
            state.conditions.propulsion.motor_torque              [N-m]
            state.conditions.propulsion.battery_state_of_charge   [None]
            state.conditions.propulsion.battery_thevenin_voltage  [V]
            state.conditions.propulsion.propeller_torque          [N-m]
            state.conditions.propulsion.battery_cell_temperature  [K] 
            state.unknowns.battery_state_of_charge                [None]
            state.unknowns.battery_cell_temperature               [K] 
            state.unknowns.battery_thevenin_voltage               [V]   
            

            Outputs:
            segment.state.residuals

            Properties Used:
            N/A
        """        
        # Unpack
        q_motor      = segment.state.conditions.propulsion.motor_torque
        q_prop       = segment.state.conditions.propulsion.propeller_torque 
        
        SOC_actual   = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict  = segment.state.unknowns.battery_state_of_charge 
        
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
        v_th_actual  = segment.state.conditions.propulsion.battery_thevenin_voltage
        v_th_predict = segment.state.unknowns.battery_thevenin_voltage        
       
        # Return the residuals 
        segment.state.residuals.network[:,0] = q_motor[:,0] - q_prop[:,0] 
        segment.state.residuals.network[:,1] = v_th_predict[:,0] - v_th_actual[:,0]     
        segment.state.residuals.network[:,2] = SOC_predict[:,0] - SOC_actual[:,0]  
        segment.state.residuals.network[:,3] = Temp_predict[:,0] - Temp_actual[:,0]
        
        return 
    
    
    def unpack_unknowns_linca_charge(self,segment):   
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.

            Assumptions:
            None 

            Source:
            N/A
            
            Inputs:
            state.unknowns.battery_state_of_charge                 [None]
            state.unknowns.battery_cell_temperature                [K]
            state.unknowns.battery_thevenin_voltage                [V]
 
            Outputs: 
            state.conditions.propulsion.battery_state_of_charge    [None]
            state.conditions.propulsion.battery_cell_temperature   [K]
            state.conditions.propulsion.battery_thevenin_voltage   [V]

            Properties Used:
            N/A
        """      
        segment.state.conditions.propulsion.battery_cell_temperature    = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge     = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_thevenin_voltage    = segment.state.unknowns.battery_thevenin_voltage  
  
        return
    
    def residuals_linca_charge(self,segment):  
        """ This packs the residuals to be send to the mission solver.

           Assumptions:
           None

           Source:
           N/A

           Inputs:
           state.conditions.propulsion.battery_state_of_charge    [None]
           state.conditions.propulsion.battery_cell_temperature   [K]
           state.conditions.propulsion.battery_thevenin_voltage   [V] 
           state.unknowns.battery_state_of_charge                 [None]
           state.unknowns.battery_cell_temperature                [K]
           state.unknowns.battery_thevenin_voltage                [V]

           Outputs:
           segment.state.residuals

           Properties Used:
           N/A
        """         
        # Unpack         
        SOC_actual   = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict  = segment.state.unknowns.battery_state_of_charge 
        
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
        v_th_actual  = segment.state.conditions.propulsion.battery_thevenin_voltage
        v_th_predict = segment.state.unknowns.battery_thevenin_voltage        
       
        # Return the residuals  
        segment.state.residuals.network[:,0] = v_th_predict[:,0] - v_th_actual[:,0]     
        segment.state.residuals.network[:,1] = SOC_predict[:,0] - SOC_actual[:,0]  
        segment.state.residuals.network[:,2] = Temp_predict[:,0] - Temp_actual[:,0]
        
        return 
    
    # -----------------------------------------------------------------
    # LiMnCO Battery Cell Unknows and Residuals 
    # -----------------------------------------------------------------    
    def unpack_unknowns_linmco(self,segment): 
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.

            Assumptions:
            None

            Source:
            N/A

            Inputs: 
            state.unknowns.battery_state_of_charge                 [None]
            state.unknowns.battery_cell_temperature                [K]
            state.unknowns.battery_current                         [A]
            
            Outputs:  
            state.conditions.propulsion.battery_state_of_charge    [None]
            state.conditions.propulsion.battery_cell_temperature   [K]
            state.conditions.propulsion.battery_current            [A]
            

            Properties Used:
            N/A
        """                
        segment.state.conditions.propulsion.propeller_power_coefficient = segment.state.unknowns.propeller_power_coefficient        
        segment.state.conditions.propulsion.battery_cell_temperature    = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge     = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_current             = segment.state.unknowns.battery_current  
      
        return
    
    def residuals_linmco(self,segment):  
        """ This packs the residuals to be send to the mission solver.
            Only the lift motors are operational
    
            Assumptions:
            The lift motors are operational
    
            Source:
            N/A
    
            Inputs: 
            state.conditions.propulsion.motor_torque              [N-m]
            state.conditions.propulsion.rotor_torque              [N-m]
            state.conditions.propulsion.battery_state_of_charge   [None]
            state.conditions.propulsion.battery_cell_temperature  [K]
            state.conditions.propulsion.battery_current           [A]
            state.unknowns.battery_state_of_charge                [None]
            state.unknowns.battery_cell_temperature               [K]
            state.unknowns.battery_current                        [A]
            
            Outputs:
            segment.state.residuals
    
            Properties Used: 
            N/A 
        """                      
        # Unpack       
        q_motor      = segment.state.conditions.propulsion.motor_torque
        q_prop       = segment.state.conditions.propulsion.propeller_torque  
        
        SOC_actual  = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict = segment.state.unknowns.battery_state_of_charge 

        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   

        i_actual     = segment.state.conditions.propulsion.battery_current
        i_predict    = segment.state.unknowns.battery_current      
      
        # Return the residuals  
        segment.state.residuals.network[:,0] =  q_motor[:,0] - q_prop[:,0]     
        segment.state.residuals.network[:,1] =  SOC_predict[:,0]  - SOC_actual[:,0]  
        segment.state.residuals.network[:,2] =  Temp_predict[:,0] - Temp_actual[:,0]
        segment.state.residuals.network[:,3] =  i_predict[:,0] - i_actual[:,0]       
         
        return 

    def unpack_unknowns_linmco_charge(self,segment): 
        """ This is an extra set of unknowns which are unpacked from the mission solver and send to the network.

            Assumptions:
            None

            Source:
            N/A

            Inputs: 
            state.unknowns.battery_state_of_charge                 [None]
            state.unknowns.battery_cell_temperature                [K]
            state.unknowns.battery_current                         [A]
            
            Outputs:  
            state.conditions.propulsion.battery_state_of_charge    [None]
            state.conditions.propulsion.battery_cell_temperature   [K]
            state.conditions.propulsion.battery_current            [A]
            

            Properties Used:
            N/A
        """          
        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_current          = segment.state.unknowns.battery_current         
      
        return
    
    def residuals_linmco_charge(self,segment):  
        """ This packs the residuals to be send to the mission solver.

            Assumptions:
            None

            Source:
            N/A

            Inputs:
            state.conditions.propulsion.battery_state_of_charge    [None]
            state.conditions.propulsion.battery_cell_temperature   [K]
            state.conditions.propulsion.battery_thevenin_voltage   [V] 
            state.unknowns.battery_state_of_charge                 [None]
            state.unknowns.battery_cell_temperature                [K]
            state.unknowns.battery_thevenin_voltage                [V]

            Outputs:
            segment.state.residuals

            Properties Used: 
            N/A
        """        
               
        # Unpack 
        SOC_actual  = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict = segment.state.unknowns.battery_state_of_charge 

        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   

        i_actual     = segment.state.conditions.propulsion.battery_current
        i_predict    = segment.state.unknowns.battery_current      

        # Return the residuals 
        segment.state.residuals.network[:,0] =  i_predict[:,0] - i_actual[:,0]    
        segment.state.residuals.network[:,1] =  SOC_predict[:,0]  - SOC_actual[:,0]  
        segment.state.residuals.network[:,2] =  Temp_predict[:,0] - Temp_actual[:,0]

        
        return 
    
    __call__ = evaluate_thrust

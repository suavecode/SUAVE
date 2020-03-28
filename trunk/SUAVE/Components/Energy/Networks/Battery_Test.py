## @ingroup Components-Energy-Networks
# Battery_Test.py
# 
# Created: Oct 2019, 2019
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
class Battery_Test(Propulsor):
    """ This is a test bench for new batteries 
    
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
        self.avionics                = None
        self.voltage                 = None 
        self.dischage_model_fidelity = 1
        
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
            conditions.propulsion: 
                current              [amps]
                battery_draw         [watts]
                battery_energy       [joules]
                voltage_open_circuit [volts]
                voltage_under_load   [volts]  
    
            Properties Used:
            Defaulted values
        """          
    
        # unpack
        conditions        = state.conditions
        numerics          = state.numerics 
        avionics          = self.avionics 
        battery           = self.battery   
        D                 = numerics.time.differentiate    
        I                 = numerics.time.integrate 
        battery_data      = battery.discharge_performance_map 
        
        # Set battery energy
        battery.current_energy      = conditions.propulsion.battery_energy  
        battery.temperature         = conditions.propulsion.battery_temperature
        battery.charge_throughput   = conditions.propulsion.battery_charge_throughput
        battery.ambient_temperature = conditions.propulsion.ambient_temperature          
        battery.age_in_days         = conditions.propulsion.battery_age_in_days 
        discharge_flag              = conditions.propulsion.battery_discharge   
        R_growth_factor             = conditions.propulsion.battery_resistance_growth_factor
        E_growth_factor             = conditions.propulsion.battery_capacity_fade_factor   
        battery.R_growth_factor     = R_growth_factor
        battery.E_growth_factor     = E_growth_factor
        
        #-------------------------------------------------------------------------------
        # PREDICT
        #-------------------------------------------------------------------------------        
        if battery.chemistry == 'LiNCA':  
            n_series   = battery.module_config[0]  
            n_parallel = battery.module_config[1]
            n_total    = n_series * n_parallel  
            
            SOC    = state.unknowns.battery_state_of_charge 
            T_cell = state.unknowns.battery_cell_temperature 
            V_Th   = state.unknowns.battery_thevenin_voltage/n_series            
            battery.cell_temperature = T_cell   
            
            if np.isnan(SOC ).any():
                raise AssertionError('Error')
            
            if np.isnan(T_cell ).any():
                raise AssertionError('Error')
            
            # look up tables  
            V_oc = np.zeros_like(SOC)
            R_Th = np.zeros_like(SOC)  
            C_Th = np.zeros_like(SOC)  
            R_0  = np.zeros_like(SOC)
            SOC[SOC<0] = 0
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
            SOC             = state.unknowns.battery_state_of_charge
            DOD             = 1 - SOC
            DOD[DOD<0.0]    = 0. 
            DOD[DOD>1.0]    = 1.
            DOD[np.isnan(DOD)] = 0.5
            
            T_cell                   = state.unknowns.battery_cell_temperature 
            T_cell[np.isnan(T_cell)] = 30.0
            T0_cell                  = battery.temperature             
            battery.cell_temperature = T_cell   
            T_cell[T_cell<0.0]       = 0. 
            T_cell[T_cell>50.0]      = 50.
             
            if discharge_flag:
                I_cell                = state.unknowns.battery_current  
                I_cell[I_cell<2.0]    = 2.0
                I_cell[np.isnan(I_cell)] = 5.0
                I_cell[I_cell>8.0]    = 8.0
            else:
                I_cell      = battery.charging_current * np.ones_like(DOD)
            
            pts    = np.hstack((np.hstack((I_cell, T_cell)), DOD )) # amps, temp, SOC 
            volts  = np.atleast_2d(battery_data.Voltage(pts)[:,1]).T 
          
        #-------------------------------------------------------------------------------
        # DISCHARGE
        #-------------------------------------------------------------------------------
        if discharge_flag:
            # Calculate avionics and payload power
            avionics_power = np.ones((numerics.number_control_points,1))*avionics.current * volts
    
            # Calculate avionics and payload current
            avionics_current =  np.ones((numerics.number_control_points,1))*avionics.current    
            
            # link
            battery.inputs.current  = avionics_current
            battery.inputs.power_in = -avionics_power
            battery.inputs.voltage  = volts
            battery.energy_discharge(numerics)          
            
        else: 
            
            # link 
            battery.inputs.current  = -battery.charging_current * np.ones_like(volts)
            battery.inputs.power_in =  battery.charging_current * volts * np.ones_like(volts)
            battery.inputs.voltage  = volts
            battery.energy_charge(numerics)        
    
        # Pack the conditions for outputs   
        battery_draw             = battery.inputs.power_in 
        battery_energy           = battery.current_energy
        state_of_charge          = battery.state_of_charge  
        
        voltage_open_circuit     = battery.voltage_open_circuit
        voltage_under_load       = battery.voltage_under_load  
        cell_temperature         = battery.cell_temperature
        
        if battery.chemistry == 'LiNCA':   
            conditions.propulsion.battery_thevenin_voltage               = battery.battery_thevenin_voltage 
            
        battery_charge_throughput= battery.charge_throughput
        current                  = battery.current 
        battery_age_in_days      = battery.age_in_days
        
        conditions.propulsion.battery_current                        = abs(current)
        conditions.propulsion.battery_draw                           = battery_draw
        conditions.propulsion.battery_energy                         = battery_energy
        conditions.propulsion.battery_charge_throughput              = battery_charge_throughput 
        conditions.propulsion.battery_OCV                            = voltage_open_circuit 
        conditions.propulsion.battery_state_of_charge                = state_of_charge
        conditions.propulsion.voltage_open_circuit                   = voltage_open_circuit
        conditions.propulsion.voltage_under_load                     = voltage_under_load    
        conditions.propulsion.battery_age_in_days                    = battery_age_in_days
        conditions.propulsion.battery_cell_temperature               = cell_temperature
        conditions.propulsion.battery_specfic_power                  = -(battery_draw/1000)/battery.mass_properties.mass   
        
        return  
    
    def unpack_unknowns_linmco_discharge(self,segment): 
        
        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_current  = segment.state.unknowns.battery_current 
        
        return
    
    def residuals_linmco_discharge(self,segment):  
        # Unpack 
        SOC_actual  = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict = segment.state.unknowns.battery_state_of_charge 
         
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
        I_actual  = segment.state.conditions.propulsion.battery_current
        I_predict = segment.state.unknowns.battery_current        
        
        # Return the residuals 
        segment.state.residuals.network[:,0] =  I_predict[:,0] - I_actual[:,0]     
        segment.state.residuals.network[:,1] =  SOC_predict[:,0]  - SOC_actual[:,0]  
        segment.state.residuals.network[:,2] =  Temp_predict[:,0] - Temp_actual[:,0]

        
        return   
    
    
    def unpack_unknowns_linmco_charge(self,segment): 

        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge

        return

    def residuals_linmco_charge(self,segment):  
        # Unpack 
        SOC_actual  = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict = segment.state.unknowns.battery_state_of_charge 

        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   

        # Return the residuals 
        segment.state.residuals.network[:,0] =  SOC_predict[:,0]  - SOC_actual[:,0]  
        segment.state.residuals.network[:,1] =  Temp_predict[:,0] - Temp_actual[:,0]


        return   
    


    def unpack_unknowns_thevenin(self,segment): 
        
        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_thevenin_voltage = segment.state.unknowns.battery_thevenin_voltage  
        
        return
    
    def residuals_thevenin(self,segment):   
        # Unpack 
        SOC_actual  = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict = segment.state.unknowns.battery_state_of_charge 
         
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
        
        v_th_actual  = segment.state.conditions.propulsion.battery_thevenin_voltage
        v_th_predict = segment.state.unknowns.battery_thevenin_voltage        
        
        # Return the residuals 
        segment.state.residuals.network[:,0] =  v_th_predict[:,0] - v_th_actual[:,0]     
        segment.state.residuals.network[:,1] =  SOC_predict[:,0]  - SOC_actual[:,0]  
        segment.state.residuals.network[:,2] =  Temp_predict[:,0] - Temp_actual[:,0]
                    
    __call__ = evaluate_thrust



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
from SUAVE.Methods.Power.Battery.Discharge.thevenin_discharge import battery_performance_maps
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
        time              = numerics.time.control_points
        D                 = numerics.time.differentiate    
        I                 = numerics.time.integrate
        dischage_fidelity = self.dischage_model_fidelity 
        battery_data      = battery_performance_maps() 
        
        # Set battery energy
        battery.current_energy    = conditions.propulsion.battery_energy  
        battery.temperature       = conditions.propulsion.battery_temperature
        battery.charge_throughput = conditions.propulsion.battery_charge_throughput
        battery.age_in_days       = conditions.propulsion.battery_age_in_days 
        discharge_flag            = conditions.propulsion.battery_discharge    

        R_growth_factor           = conditions.propulsion.battery_resistance_growth_factor
        E_growth_factor           = conditions.propulsion.battery_capacity_growth_factor  
        
        battery.R_growth_factor   = R_growth_factor
        battery.E_growth_factor   = E_growth_factor
        
        if dischage_fidelity == 1: 
            volts = state.unknowns.battery_voltage_under_load
            battery.battery_thevenin_voltage = 0 
            battery.cell_temperature = battery.temperature
            
        elif dischage_fidelity == 2:    
            SOC             = state.unknowns.battery_state_of_charge 
            T_cell          = state.unknowns.battery_cell_temperature 
            V_Th            = state.unknowns.battery_thevenin_voltage             
            Q_prior         = battery.charge_throughput 
            t               = battery.age_in_days 
            
            battery.cell_temperature = T_cell 
            
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
                R_0[i]  =  battery_data.R_0_interp(T_cell[i], SOC[i])[0]
            
            dV_TH_dt =  np.dot(D,V_Th)
            I_0 = V_Th/R_Th  + C_Th*dV_TH_dt 
            R_0  = R_0 * R_growth_factor  
        
            # Voltage under load:
            volts =  V_oc - V_Th - (I_0 * R_0)        
            
        if discharge_flag:
            # Calculate avionics and payload power
            avionics_power = np.ones((numerics.number_control_points,1))*avionics.current * volts
    
            # Calculate avionics and payload current
            avionics_current =  np.ones((numerics.number_control_points,1))*avionics.current    
            
            # link
            battery.inputs.current  = avionics_current
            battery.inputs.power_in = -avionics_power
            battery.inputs.V_ul     = volts 
            battery.energy_discharge(numerics,fidelity = dischage_fidelity)          
            
        else: 

            # link 
            battery.inputs.current  = -battery.charging_current * np.ones_like(volts)
            battery.inputs.power_in =  battery.charging_current * volts * np.ones_like(volts)
            battery.inputs.V_ul     =  volts 
        
            battery.energy_charge(numerics,fidelity = dischage_fidelity)        
    
        # Pack the conditions for outputs   
        battery_draw             = battery.inputs.power_in 
        battery_energy           = battery.current_energy
        state_of_charge          = battery.state_of_charge  
        voltage_open_circuit     = battery.voltage_open_circuit
        voltage_under_load       = battery.voltage_under_load  
        cell_temperature         = battery.cell_temperature
        battery_thevenin_voltage = battery.battery_thevenin_voltage
        battery_charge_throughput= battery.charge_throughput
        current                  = battery.current 
        battery_age_in_days      = battery.age_in_days
        
        conditions.propulsion.current                                = abs(current)
        conditions.propulsion.battery_draw                           = battery_draw
        conditions.propulsion.battery_energy                         = battery_energy
        conditions.propulsion.battery_charge_throughput              = battery_charge_throughput   
        conditions.propulsion.state_of_charge                        = state_of_charge
        conditions.propulsion.voltage_open_circuit                   = voltage_open_circuit
        conditions.propulsion.voltage_under_load                     = voltage_under_load    
        conditions.propulsion.battery_thevenin_voltage               = battery_thevenin_voltage
        conditions.propulsion.battery_age_in_days                    = battery_age_in_days
        conditions.propulsion.battery_cell_temperature               = cell_temperature
        conditions.propulsion.battery_specfic_power                  = -(battery_draw/1000)/battery.mass_properties.mass   
        
        return  
    
    def unpack_unknowns_datta(self,segment): 
        
        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_voltage_under_load  = segment.state.unknowns.battery_voltage_under_load
        
        return
    
    def residuals_datta(self,segment):  
        # Unpack 
        v_actual     = segment.state.conditions.propulsion.voltage_under_load
        v_predict    = segment.state.unknowns.battery_voltage_under_load
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature           
        
        # Return the residuals 
        segment.state.residuals.network[:,0] = v_predict[:,0] - v_actual[:,0]
        segment.state.residuals.network[:,1] = Temp_predict[:,0] - Temp_actual[:,0]
        
        return   
    
    
    def unpack_unknowns_thevenin(self,segment): 
        
        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_thevenin_voltage = segment.state.unknowns.battery_thevenin_voltage  
        
        return
    
    def residuals_thevenin(self,segment):   
        # Unpack 
        SOC_actual  = segment.state.conditions.propulsion.state_of_charge
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



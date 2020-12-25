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
        self.tag                     = 'Battery_Test'

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
                current                       [amps]
                battery_power_draw            [watts]
                battery_energy                [joules]
                battery_voltage_open_circuit  [volts]
                battery_voltage_under_load    [volts]  
    
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
        battery.pack_temperature    = conditions.propulsion.battery_pack_temperature
        battery.charge_throughput   = conditions.propulsion.battery_cumulative_charge_throughput
        battery.ambient_temperature = conditions.propulsion.ambient_temperature          
        battery.charge_throughput   = conditions.propulsion.battery_cumulative_charge_throughput
        battery.age_in_days         = conditions.propulsion.battery_age_in_days
        discharge_flag              = conditions.propulsion.battery_discharge
        battery.R_growth_factor     = conditions.propulsion.battery_resistance_growth_factor
        battery.E_growth_factor     = conditions.propulsion.battery_capacity_fade_factor
        battery.max_energy          = conditions.propulsion.battery_max_aged_energy
        V_th0                       = conditions.propulsion.battery_initial_thevenin_voltage
        n_series                    = battery.pack_config.series  
        n_parallel                  = battery.pack_config.parallel
         
        #-------------------------------------------------------------------------------
        # Predict Voltage and Battery Properties Depending on Battery Chemistry
        #-------------------------------------------------------------------------------        
        if battery.chemistry == 'LiNCA':
            # update ambient temperature based on altitude
            battery.ambient_temperature                   = conditions.propulsion.ambient_temperature
            battery.cooling_fluid.thermal_conductivity    = conditions.freestream.thermal_conductivity
            battery.cooling_fluid.kinematic_viscosity     = conditions.freestream.kinematic_viscosity
            battery.cooling_fluid.density                 = conditions.freestream.density

            SOC        = state.unknowns.battery_state_of_charge 
            T_cell     = state.unknowns.battery_cell_temperature 
            V_Th_cell  = state.unknowns.battery_thevenin_voltage/n_series
            
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
            volts = n_series*(V_oc_cell - V_Th_cell - (I_cell * R_0_cell))
            
            battery.charging_current = battery.cell.charging_current * n_parallel 
            
        elif battery.chemistry == 'LiNiMnCoO2': 
            # update ambient temperature based on altitude
            battery.ambient_temperature                   = conditions.propulsion.ambient_temperature
            battery.cooling_fluid.thermal_conductivity    = conditions.freestream.thermal_conductivity
            battery.cooling_fluid.kinematic_viscosity     = conditions.freestream.kinematic_viscosity 
            battery.cooling_fluid.density                 = conditions.freestream.density

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
            
            T_cell[T_cell<272.65]    = 272.65 # model does not fit for below 0  degrees
            T_cell[T_cell>322.65]    = 322.65 # model does not fit for above 50 degrees
             
            I_cell[I_cell<0.0]       = 0.0
            I_cell[I_cell>8.0]       = 8.0    
            
            # create vector of conditions for battery data sheet response surface for OCV
            T_cell_Celcius           = T_cell-272.65
            pts                      = np.hstack((np.hstack((I_cell, T_cell_Celcius)),DOD  )) # amps, temp, SOC
            V_ul_cell                = np.atleast_2d(battery_data.Voltage(pts)[:,1]).T
            volts                    = n_series*V_ul_cell
          
            battery.charging_current = battery.cell.charging_current * n_parallel 
            
        else:
            battery.cell_temperature         = conditions.propulsion.battery_cell_temperature
            volts                            = state.unknowns.battery_voltage_under_load
            battery.battery_thevenin_voltage = 0

        #-------------------------------------------------------------------------------
        # Discharge
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
            battery.inputs.power_in =  battery.charging_current * battery.charging_voltage * np.ones_like(volts)
            battery.inputs.voltage  =  battery.charging_voltage 
            battery.energy_charge(numerics)        
        
        # Pack the conditions for outputs     
        conditions.propulsion.battery_thevenin_voltage             = battery.thevenin_voltage 
        conditions.propulsion.battery_current                      = abs( battery.current )
        conditions.propulsion.battery_power_draw                   = battery.inputs.power_in 
        conditions.propulsion.battery_energy                       = battery.current_energy  
        conditions.propulsion.battery_cumulative_charge_throughput = battery.cumulative_cell_charge_throughput 
        conditions.propulsion.battery_state_of_charge              = battery.state_of_charge
        conditions.propulsion.battery_voltage_open_circuit         = battery.voltage_open_circuit
        conditions.propulsion.battery_voltage_under_load           = battery.voltage_under_load  
        conditions.propulsion.battery_internal_resistance          = battery.internal_resistance
        conditions.propulsion.battery_age_in_days                  = battery.age_in_days
        conditions.propulsion.battery_pack_temperature             = battery.pack_temperature
        conditions.propulsion.battery_cell_temperature             = battery.cell_temperature
        conditions.propulsion.battery_cell_joule_heat_fraction     = battery.cell_joule_heat_fraction   
        conditions.propulsion.battery_cell_entropy_heat_fraction   = battery.cell_entropy_heat_fraction        
        conditions.propulsion.battery_specfic_power                = -(battery.inputs.power_in /1000)/battery.mass_properties.mass   
          
        F     = np.zeros_like(volts)  * [0,0,0]      
        mdot  = state.ones_row(1)*0.0
         
        results                     = Data()
        results.thrust_force_vector = F
        results.vehicle_mass_rate   = mdot  
        
        return results 
    

    def unpack_unknowns_linmco(self,segment):         
        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_current          = segment.state.unknowns.battery_current  

        return

    def residuals_linmco(self,segment):  
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

    def unpack_unknowns_linca(self,segment): 

        segment.state.conditions.propulsion.battery_cell_temperature = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge  = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_thevenin_voltage = segment.state.unknowns.battery_thevenin_voltage  

        return

    def residuals_linca(self,segment):   
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



## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion_LiNCA_18650.py
# 
# Created:  Feb 2020, M. Clarke
# Modified: Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
from SUAVE.Core      import Units , Data 
from .Lithium_Ion    import Lithium_Ion 
from SUAVE.Methods.Power.Battery.Cell_Cycle_Models.LiNCA_cell_cycle_model import compute_NCA_cell_state_variables
from SUAVE.Methods.Power.Battery.compute_net_generated_battery_heat       import compute_net_generated_battery_heat

import os
import numpy as np   
from scipy.integrate   import  cumtrapz  
from scipy.interpolate import RectBivariateSpline

## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion_LiNCA_18650(Lithium_Ion): 
    """ Specifies discharge/specific energy characteristics specific 
        18650 lithium-nickel-cobalt-aluminum oxide (LiNCA) battery cells.  
        
        Assumptions:
        Convective Thermal Conductivity Coefficient corresponds to forced
        air cooling in 35 m/s air 
        
        Source:
        Discharge Information
        Intriduction of INR18650-30Q. https://eu.nkon.nl/sk/k/30q.pdf
        
        Convective  Heat Transfer Coefficient, h 
        Jeon, Dong Hyup, and Seung Man Baek. "Thermal modeling of cylindrical 
        lithium ion battery during discharge cycle." Energy Conversion and Management
        52.8-9 (2011): 2973-2981.
        
        Thermal Conductivity, k 
        (radial)
        Murashko, Kirill A., Juha Pyrhönen, and Jorma Jokiniemi. "Determination of the 
        through-plane thermal conductivity and specific heat capacity of a Li-ion cylindrical 
        cell." International Journal of Heat and Mass Transfer 162 (2020): 120330.
         
        Specific Heat Capacity, Cp
        Yang, Shuting, et al. "A Review of Lithium-Ion Battery Thermal Management 
        System Strategies and the Evaluate Criteria." Int. J. Electrochem. Sci 14
        (2019): 6077-6107. 
        
        Inputs:
        None
        
        Outputs:
        None
        
        Properties Used:
        N/A
        
    """   
    def __defaults__(self):    
        self.tag                              = 'Lithium_Ion_LiNCA_Cell' 
                                             
        self.cell.diameter                    = 0.0185                                                   # [m]
        self.cell.height                      = 0.0653                                                   # [m]
        self.cell.mass                        = 0.048 * Units.kg                                         # [kg]
        self.cell.surface_area                = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)  # [m^2]
        self.cell.volume                      = np.pi*(0.5*self.cell.diameter)**2*self.cell.height 
        self.cell.density                     = self.cell.mass/self.cell.volume                          # [kg/m^3]  
        self.cell.electrode_area              = 0.0342                                                   # [m^2] 
                                                                                                 
        self.cell.max_voltage                 = 4.2                                                      # [V]
        self.cell.nominal_capacity            = 3.30                                                     # [Amp-Hrs]
        self.cell.nominal_voltage             = 3.6                                                      # [V]
        self.cell.charging_voltage            = self.cell.nominal_voltage                                # [V]  
        
        self.watt_hour_rating                 = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
        self.specific_energy                  = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
        self.specific_power                   = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
        self.resistance                       = 0.025                                                    # [Ohms]
                                                                                               
        self.specific_heat_capacity           = 837.4                                                    # [J/kgK] 
        self.cell.specific_heat_capacity      = 837.4                                                    # [J/kgK]   
        self.cell.radial_thermal_conductivity = 0.8                                                      # [J/kgK]  
        self.cell.axial_thermal_conductivity  = 32.2                                                     # [J/kgK]  
            
        battery_raw_data                      = load_NCA_raw_results()                                                   
        self.discharge_performance_map        = create_discharge_performance_map(battery_raw_data)        
        return  
    
    
    def energy_calc(self,numerics,battery_discharge_flag = True ):  
        """This is an electric cycle model for 18650 lithium-nickel-cobalt-aluminum oxide battery
           using a thevenin equavalent circuit with parameters taken from 
           pulse tests done by NASA Glen (referece below) of a Samsung (SDI 18650-30Q). 
         
           Assumtions:
           1) All battery modules exhibit the same themal behaviour.
           
           Sources:   
           Entropy Model:
           Santhanagopalan, Shriram, Qingzhi Guo, and Ralph E. White. "Parameter estimation 
           and model discrimination for a lithium-ion cell." Journal of the Electrochemical 
           Society 154.3 (2007): A198. 
           
           Battery Heat Generation Model:
           Jeon, Dong Hyup, and Seung Man Baek. "Thermal modeling of cylindrical lithium ion 
           battery during discharge cycle." Energy Conversion and Management 52.8-9 (2011): 
           2973-2981. 
           
           Inputs:
             battery. 
                   I_bat             (max_energy)                          [Joules]
                   cell_mass         (battery cell mass)                   [kilograms]
                   Cp                (battery cell specific heat capacity) [J/(K kg)] 
                   t                 (battery age in days)                 [days] 
                   T_ambient         (ambient temperature)                 [Kelvin]
                   T_current         (pack temperature)                    [Kelvin]
                   T_cell            (battery cell temperature)            [Kelvin]
                   E_max             (max energy)                          [Joules]
                   E_current         (current energy)                      [Joules]
                   Q_prior           (charge throughput)                   [Amp-hrs]
                   R_growth_factor   (internal resistance growth factor)   [unitless] 
               
             inputs.
                   I_bat             (current)                             [amps]
                   P_bat             (power)                               [Watts]
           
           Outputs:
             battery.          
                  current_energy                                           [Joules]
                  cell_temperature                                         [Kelvin]
                  resistive_losses                                         [Watts] 
                  load_power                                               [Watts]
                  current                                                  [Amps]
                  battery_voltage_open_circuit                             [Volts]
                  battery_thevenin_voltage                                 [Volts]
                  cell_charge_throughput                                   [Amp-hrs]
                  internal_resistance                                      [Ohms]
                  battery_state_of_charge                                  [unitless]
                  depth_of_discharge                                       [unitless]
                  battery_voltage_under_load                               [Volts]   
            
        """
        
        # Unpack varibles 
        battery                  = self
        I_bat                    = battery.inputs.current
        P_bat                    = battery.inputs.power_in   
        cell_mass                = battery.cell.mass   
        electrode_area           = battery.cell.electrode_area
        Cp                       = battery.cell.specific_heat_capacity  
        As_cell                  = battery.cell.surface_area  
        T_current                = battery.pack_temperature      
        T_cell                   = battery.cell_temperature     
        E_max                    = battery.max_energy
        R_growth_factor          = battery.R_growth_factor 
        E_current                = battery.current_energy 
        Q_prior                  = battery.cell_charge_throughput  
        battery_data             = battery.discharge_performance_map  
        I                        = numerics.time.integrate  
        D                        = numerics.time.differentiate      
        
        # ---------------------------------------------------------------------------------
        # Compute battery electrical properties 
        # ---------------------------------------------------------------------------------
        
        # Calculate the current going into one cell  
        n_series          = battery.pack_config.series  
        n_parallel        = battery.pack_config.parallel
        n_total           = battery.pack_config.total
        Nn                = battery.module_config.normal_count            
        Np                = battery.module_config.parallel_count          
        n_total_module    = Nn*Np    
        
        if battery_discharge_flag :
            I_cell = I_bat/n_parallel
        else:  
            I_cell = -I_bat/n_parallel
        
        # State of charge of the battery
        initial_discharge_state = np.dot(I,P_bat) + E_current[0]
        SOC_old =  np.divide(initial_discharge_state,E_max)
        SOC_old[SOC_old < 0.] = 0.    
        SOC_old[SOC_old > 1.] = 1.      
        
        # ---------------------------------------------------------------------------------
        # Compute battery cell temperature 
        # ---------------------------------------------------------------------------------
        # Determine temperature increase         
        sigma   = 100   
        n       = 1
        F       = 96485  # C/mol Faraday constant    
        delta_S = -18046*(SOC_old)**6 + 52735*(SOC_old)**5 - 57196*(SOC_old)**4 + \
                   28030*(SOC_old)**3 -6023*(SOC_old)**2 +  514*(SOC_old) -27
        
        i_cell         = I_cell/electrode_area # current intensity 
        q_dot_entropy  = -(T_cell)*delta_S*i_cell/(n*F)  
        q_dot_joule    = (i_cell**2)/sigma                 
        Q_heat_gen     = (q_dot_joule + q_dot_entropy)*As_cell 
        q_joule_frac   = q_dot_joule/(q_dot_joule + q_dot_entropy)
        q_entropy_frac = q_dot_entropy/(q_dot_joule + q_dot_entropy) 
        
        # Compute net heat generated 
        P_net = compute_net_generated_battery_heat(n_total,battery,Q_heat_gen)    
        
        dT_dt     = P_net/(cell_mass*n_total_module*Cp)
        T_current = T_current[0] + np.dot(I,dT_dt)  
    
        # Power going into the battery accounting for resistance losses
        P_loss = n_total*Q_heat_gen
        P      = P_bat - np.abs(P_loss)      
        
        # Compute state variables
        V_oc,C_Th,R_Th,R_0 = compute_NCA_cell_state_variables(battery_data,SOC_old,T_cell)   
        
        # Compute thevening equivalent voltage  
        V_Th = I_cell/(1/R_Th + C_Th*np.dot(D,np.ones_like(R_Th)))
        
        # Update battery internal and thevenin resistance with aging factor 
        R_0_aged   = R_0 * R_growth_factor     
       
        # Calculate resistive losses
        Q_heat_gen = (I_cell**2)*(R_0_aged + R_Th)
          
        # Power going into the battery accounting for resistance losses
        P_loss = n_total*Q_heat_gen
        P = P_bat - np.abs(P_loss) 
        
        # ---------------------------------------------------------------------------------
        # Compute updates state of battery 
        # ---------------------------------------------------------------------------------   
        
        # Determine actual power going into the battery accounting for resistance losses
        E_bat = np.dot(I,P) 
        
        # Determine current energy state of battery (from all previous segments)          
        E_current = E_bat + E_current[0]
        E_current[E_current>E_max] = E_max
        
        # Determine new State of Charge 
        SOC_new = np.divide(E_current, E_max)
        SOC_new[SOC_new<0] = 0. 
        SOC_new[SOC_new>1] = 1.
        DOD_new = 1 - SOC_new
        
        # Determine voltage under load:
        V_ul   = V_oc - V_Th - (I_cell * R_0_aged)
         
        # Determine new charge throughput (the amount of charge gone through the battery)
        Q_total    = np.atleast_2d(np.hstack(( Q_prior[0] , Q_prior[0] + cumtrapz(I_cell[:,0], x = numerics.time.control_points[:,0])/Units.hr ))).T   
      
        # If SOC is negative, voltage under load goes to zero 
        V_ul[SOC_new < 0.] = 0.
        
        # Pack outputs
        battery.current_energy                      = E_current
        battery.cell_temperature                    = T_current
        battery.pack_temperature                    = T_current
        battery.resistive_losses                    = P_loss
        battery.load_power                          = V_ul*n_series*I_bat
        battery.current                             = I_bat 
        battery.voltage_open_circuit                = V_oc*n_series
        battery.thevenin_voltage                    = V_Th*n_series 
        battery.cell_joule_heat_fraction            = q_joule_frac
        battery.cell_entropy_heat_fraction          = q_entropy_frac 
        battery.cell_charge_throughput              = Q_total
        battery.internal_resistance                 = R_0*n_series 
        battery.state_of_charge                     = SOC_new
        battery.depth_of_discharge                  = DOD_new
        battery.voltage_under_load                  = V_ul*n_series  
        battery.cell_voltage_open_circuit           = V_oc
        battery.cell_current                        = I_cell 
        battery.heat_energy_generated               = Q_heat_gen*n_total_module 
        battery.cell_voltage_under_load             = V_ul
        battery.cell_joule_heat_fraction            = np.zeros_like(V_ul)
        battery.cell_entropy_heat_fraction          = np.zeros_like(V_ul)
        
        return battery   
    
    def append_battery_unknowns(self,segment): 
        """ Appends unknowns specific to NCA cells which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment.state.unknowns.battery_cell_temperature   [Kelvin]
            segment.state.unknowns.battery_state_of_charge    [unitless]
            segment.state.unknowns.battery_thevenin_voltage   [volts]
    
            Outputs: 
            segment.state.conditions.propulsion.battery_cell_temperature  [Kelvin]  
            segment.state.conditions.propulsion.battery_state_of_charge   [unitless]
            segment.state.conditions.propulsion.battery_thevenin_voltage  [volts]
    
            Properties Used:
            N/A
        """    
     
        
        propulsion = segment.state.conditions.propulsion
        
        propulsion.battery_cell_temperature[1:,0] = segment.state.unknowns.battery_cell_temperature[:,0]
        propulsion.battery_state_of_charge[1:,0]  = segment.state.unknowns.battery_state_of_charge[:,0]
        propulsion.battery_thevenin_voltage       = segment.state.unknowns.battery_thevenin_voltage   
        
        return     

    def append_battery_residuals(self,segment,network): 
        """ Packs the residuals specific to NCA cells to be sent to the mission solver.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment.state.conditions.propulsion:
                battery_state_of_charge      [unitless] 
                battery_cell_temperature     [Kelvin]        
                battery_thevenin_voltage     [Volts]
            segment.state.unknowns.
                battery_state_of_charge      [unitless]
                battery_cell_temperature     [Kelvin]  
                battery_thevenin_voltage     [Volts]
            Outputs:
            None
    
            Properties Used:
            None
        """      
        
        SOC_actual   = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict  = segment.state.unknowns.battery_state_of_charge 
    
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
    
        v_th_actual  = segment.state.conditions.propulsion.battery_thevenin_voltage
        v_th_predict = segment.state.unknowns.battery_thevenin_voltage
        
    
        # Return the residuals   
        segment.state.residuals.network.thevenin_voltage = v_th_predict - v_th_actual  
        segment.state.residuals.network.SOC              = SOC_predict  - SOC_actual[1:,:]  
        segment.state.residuals.network.temperature      = Temp_predict - Temp_actual[1:,:]  
        
        return 
    
    
    def append_battery_unknowns_and_residuals_to_segment(self,segment,initial_voltage, 
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current,initial_battery_cell_thevenin_voltage): 
        """ Sets up the information that the mission needs to run a mission segment using this network
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:  
            initial_voltage                       [volts]
            initial_battery_cell_temperature      [Kelvin]
            initial_battery_state_of_charge       [unitless]
            initial_battery_cell_current          [Amperes]
            initial_battery_cell_thevenin_voltage [Volts]
            
            Outputs
            None
            
            Properties Used:
            N/A
        """       
        # setup the state
        ones_row    = segment.state.unknowns.ones_row
        ones_row_m1 = segment.state.unknowns.ones_row_m1
      
        segment.state.unknowns.battery_state_of_charge  = initial_battery_state_of_charge       * ones_row_m1(1)  
        segment.state.unknowns.battery_cell_temperature = initial_battery_cell_temperature      * ones_row_m1(1)        
        segment.state.unknowns.battery_thevenin_voltage = initial_battery_cell_thevenin_voltage * ones_row(1)    
        
        return  

    def compute_voltage(self,state):  
        """ Computes the voltage of a single NCA cell or a battery pack of NCA cells   
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:  
                self    - battery data structure             [unitless]
                state   - segment unknowns to define voltage [unitless]
            
            Outputs
                V_ul    - under-load voltage                 [volts]
             
            Properties Used:
            N/A
        """            
        
        # Unpack battery properties
        battery           = self
        battery_data      = battery.discharge_performance_map
        n_series          = battery.pack_config.series  
       
        # Unpack segment state properties 
        SOC       = state.conditions.propulsion.battery_state_of_charge
        T_cell    = state.conditions.propulsion.battery_cell_temperature
        V_Th_cell = state.unknowns.battery_thevenin_voltage/n_series
        D         = state.numerics.time.differentiate  
        
        # link temperature 
        battery.cell_temperature = T_cell   
         
        # Compute state variables
        V_oc_cell,C_Th_cell,R_Th_cell,R_0_cell = compute_NCA_cell_state_variables(battery_data,SOC,T_cell) 
        dV_TH_dt =  np.dot(D,V_Th_cell)
        I_cell   = V_Th_cell/(R_Th_cell * battery.R_growth_factor)  + C_Th_cell*dV_TH_dt
        R_0_cell = R_0_cell * battery.R_growth_factor
         
        # Voltage under load
        V_ul =  n_series*(V_oc_cell - V_Th_cell - (I_cell  * R_0_cell)) 
        
        return V_ul 
 
    def update_battery_state_of_health(self,segment,increment_battery_cycle_day = False):     
        print(' No aging model currently implemented for NCA cells. Pristine condition of \n '
              'the battery cell will be assigned each charge cycle')        
        return   

def create_discharge_performance_map(battery_raw_data):
    """ Creates discharge and charge response surfaces for 
        LiNCA battery cells using raw data     
        
        Source:
        N/A
        
        Assumptions:
        N/A
        
        Inputs: 
            
        Outputs: 
        battery_data

        Properties Used:
        N/A
                                
    """  
    battery_data             = Data()
    SMOOTHING                = 0.1  
    battery_data.V_oc_interp = RectBivariateSpline(battery_raw_data.T_bp, battery_raw_data.SOC_bp, battery_raw_data.tV_oc, s=SMOOTHING)  
    battery_data.C_Th_interp = RectBivariateSpline(battery_raw_data.T_bp, battery_raw_data.SOC_bp, battery_raw_data.tC_Th, s=SMOOTHING)  
    battery_data.R_Th_interp = RectBivariateSpline(battery_raw_data.T_bp, battery_raw_data.SOC_bp, battery_raw_data.tR_Th, s=SMOOTHING)  
    battery_data.R_0_interp  = RectBivariateSpline(battery_raw_data.T_bp, battery_raw_data.SOC_bp, battery_raw_data.tR_0,  s=SMOOTHING) 
 
    return battery_data


def load_NCA_raw_results(): 
    '''Load experimental raw data of NCA cells 
    
    Source:
    Intriduction of INR18650-30Q. https://eu.nkon.nl/sk/k/30q.pdf
    
    Assumptions:
    N/A
    
    Inputs: 
    N/A
        
    Outputs: 
    battery_data

    Properties Used:
    N/A  
    '''
    ospath    = os.path.abspath(__file__)
    separator = os.path.sep
    rel_path  = os.path.dirname(ospath) + separator     
    return SUAVE.Input_Output.SUAVE.load(rel_path+'NCA_Raw_Data.res') 

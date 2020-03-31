## @ingroup Methods-Power-Battery-Discharge
# LiNCA_discharge.py
# 
# Created: Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units 
import numpy as np
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import  cumtrapz 

def LiNCA_discharge (battery,numerics): 
    """his is a discharge model for lithium-nickel-cobalt-aluminum oxide 18650 battery
       using a thevenin equavalent circuit with parameters taken from 
       pulse tests done by NASA Glen (referece below) of a Samsung (SDI 18650-30Q).
       Cell Aging model was developed from fitting of experimental data of LiMNC 
       18650 cell.  
     
       Source: 
       Cell Charge: Chin, J. C., Schnulo, S. L., Miller, T. B., Prokopius, K., and Gray, 
       J., “"Battery Performance Modeling on Maxwell X-57",”AIAA Scitech, San Diego, CA,
       2019. URLhttp://openmdao.org/pubs/chin_battery_performance_x57_2019.pdf.     
       
       Cell Heat Coefficient:  Wu et. al. "Determination of the optimum heat transfer 
       coefficient and temperature rise analysis for a lithium-ion battery under 
       the conditions of Harbin city bus driving cycles". Energies, 10(11). 
       https://doi.org/10.3390/en10111723
        
       Inputs:
         battery. 
               I_bat             (max_energy)                          [Joules]
               cell_mass         (battery cell mass)                   [kilograms]
               Cp                (battery cell specific heat capacity) [J/(K kg)]
               h                 (heat transfer coefficient)           [W/(m^2*K)]
               t                 (battery age in days)                 [days]
               cell_surface_area (battery cell surface area)           [meters^2]
               T_ambient         (ambient temperature)                 [Degrees Celcius]
               T_current         (pack temperature)                    [Degrees Celcius]
               T_cell            (battery cell temperature)            [Degrees Celcius]
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
              cell_temperature                                         [Degrees Celcius]
              resistive_losses                                         [Watts] 
              load_power                                               [Watts]
              current                                                  [Amps]
              voltage_open_circuit                                     [Volts]
              battery_thevenin_voltage                                 [Volts]
              charge_throughput                                        [Amp-hrs]
              internal_resistance                                      [Ohms]
              state_of_charge                                          [unitless]
              depth_of_discharge                                       [unitless]
              voltage_under_load                                       [Volts]   
        
    """
    
    # Unpack varibles 
    I_bat             = battery.inputs.current
    P_bat             = battery.inputs.power_in   
    cell_mass         = battery.cell.mass     
    Cp                = battery.cell.specific_heat_capacity  
    t                 = battery.age_in_days
    cell_surface_area = battery.cell.surface_area
    T_ambient         = battery.ambient_temperature    
    T_current         = battery.temperature      
    T_cell            = battery.cell_temperature     
    E_max             = battery.max_energy
    E_current         = battery.current_energy 
    Q_prior           = battery.charge_throughput 
    R_growth_factor   = battery.R_growth_factor 
    battery_data      = battery.discharge_performance_map 
    I                 = numerics.time.integrate
    D                 = numerics.time.differentiate      
    
    # Calculate the current going into one cell 
    n_series   = battery.module_config[0]  
    n_parallel = battery.module_config[1]
    n_total    = n_series * n_parallel 
    I_cell     = I_bat/n_parallel
    
    # State of charge of the battery
    initial_discharge_state = np.dot(I,P_bat) + E_current[0]
    SOC_old =  np.divide(initial_discharge_state,E_max)
    SOC_old[SOC_old < 0.] = 0.    
    DOD_old = 1 - SOC_old          
    
    # Look up tables for variables as a function of temperature and SOC
    # FLAG FOR BATTERY DISCHARGE
    V_oc = np.zeros_like(I_cell)
    R_Th = np.zeros_like(I_cell)  
    C_Th = np.zeros_like(I_cell)  
    R_0  = np.zeros_like(I_cell) 
    for i in range(len(SOC_old)): 
        V_oc[i] = battery_data.V_oc_interp(T_cell[i], SOC_old[i])[0]
        C_Th[i] = battery_data.C_Th_interp(T_cell[i], SOC_old[i])[0]
        R_Th[i] = battery_data.R_Th_interp(T_cell[i], SOC_old[i])[0]
        R_0[i]  = battery_data.R_0_interp(T_cell[i], SOC_old[i])[0] 
    
    # Compute thevening equivalent voltage  
    V_Th = I_cell/(1/R_Th + C_Th*np.dot(D,np.ones_like(R_Th)))
    
    # Update battery internal and thevenin resistance with aging factor
    R_0  = R_0 * R_growth_factor
    R_Th = R_Th* R_growth_factor
   
    # Calculate resistive losses
    P_heat = (I_cell**2)*(R_0 + R_Th)
    
    # Determine temperature increase 
    h = -290 + 39.036*T_cell - 1.725*(T_cell**2) + 0.026*(T_cell**3)
    P_net      = P_heat - h*0.5*cell_surface_area*(T_cell - T_ambient) 
    dT_dt      = P_net/(cell_mass*Cp)
    T_current  = np.atleast_2d(np.hstack(( T_current[0] , T_current[0] + cumtrapz(dT_dt[:,0], x = numerics.time.control_points[:,0]) ))).T
    
    # Power going into the battery accounting for resistance losses
    P_loss = n_total*P_heat
    P = P_bat - np.abs(P_loss) 
    
    # Determine actual power going into the battery accounting for resistance losses
    E_bat = np.dot(I,P)
    
    # Add this to the current state
    if np.isnan(E_bat).any():
        E_bat=np.ones_like(E_bat)*np.max(E_bat)
        if np.isnan(E_bat.any()): #all nans; handle this instance
            E_bat = np.zeros_like(E_bat)
            
    # Determine current energy state of battery (from all previous segments)          
    E_current = E_bat + E_current[0]
    
    # Determine new State of Charge 
    SOC_new = np.divide(E_current, E_max)
    SOC_new[SOC_new<0] = 0. 
    SOC_new[SOC_new>1] = 1.
    DOD_new = 1 - SOC_new
    
    # Determine voltage under load:
    V_ul   = V_oc - V_Th - (I_cell * R_0)
    
    # Determine new charge throughput (the amount of charge gone through the battery)
    Q_total  = np.atleast_2d(np.hstack((  Q_prior[0] , Q_prior[0] + cumtrapz(I_cell[:,0], x = numerics.time.control_points[:,0])/Units.hr ))).T  
    
    # If SOC is negative, voltage under load goes to zero 
    V_ul[SOC_new < 0.] = 0.
    
    # Pack outputs
    battery.current_energy           = E_current
    battery.cell_temperature         = T_current  
    battery.resistive_losses         = P_loss
    battery.load_power               = V_ul*n_series*I_bat
    battery.current                  = I_bat
    battery.voltage_open_circuit     = V_oc*n_series
    battery.battery_thevenin_voltage = V_Th*n_series
    battery.charge_throughput        = Q_total 
    battery.internal_resistance      = R_0*n_series 
    battery.state_of_charge          = SOC_new
    battery.depth_of_discharge       = DOD_new
    battery.voltage_under_load       = V_ul*n_series 
    
    return battery

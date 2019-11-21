## @ingroup Methods-Power-Battery-Charge
# thevenin_discharge.py
# 
# Created:  Oct 2019, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units 
import numpy as np
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import odeint

def thevenin_charge(battery,numerics): 
    """Charge and Cell Ageing Model:
       Charge Model uses a thevenin equavalent circuit with parameters taken from 
       pulse tests done by NASA Glen (referece below) of a Samsung (SDI 18650-30Q).
       Cell Aging model was developed from fitting of experimental data of LiMNC 
       18650 cell.  
     
       Source: 
       Cell Charge: Chin, J. C., Schnulo, S. L., Miller, T. B., Prokopius, K., and Gray, 
       J., “"Battery Performance Modeling on Maxwell X-57",”AIAA Scitech, San Diego, CA,
       2019. URLhttp://openmdao.org/pubs/chin_battery_performance_x57_2019.pdf.
       
       Cell Aging:  Schmalstieg, Johannes, et al. "A holistic aging model for Li (NiMnCo) O2
       based 18650 lithium-ion batteries." Journal of Power Sources 257 (2014): 325-334.
       
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
               E_growth_factor   (capactance (energy) growth factor)   [unitless] 
           
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
    h                 = battery.heat_transfer_coefficient   
    t                 = battery.age_in_days
    cell_surface_area = battery.cell.surface_area
    T_ambient         = battery.ambient_temperature    
    T_current         = battery.temperature      
    T_cell            = battery.cell_temperature     
    E_max             = battery.max_energy
    E_current         = battery.current_energy 
    Q_prior           = battery.charge_throughput 
    R_growth_factor   = battery.R_growth_factor
    E_growth_factor   = battery.E_growth_factor
    battery_data      = battery_performance_maps()     
    I                 = numerics.time.integrate
    D                 = numerics.time.differentiate        
    
    # Update battery capacitance (energy) with aging factor
    E_max = E_max*E_growth_factor
    
    # Calculate the current going into one cell 
    n_series   = battery.module_config[0]  
    n_parallel = battery.module_config[1]
    n_total    = n_series * n_parallel 
    I_cell      = I_bat/n_parallel
    
    # State of charge of the battery
    initial_discharge_state = np.dot(I,P_bat) + E_current[0]
    SOC_old =  np.divide(initial_discharge_state,E_max)
    SOC_old[SOC_old < 0.] = 0.    
    DOD_old = 1 - SOC_old 
    
    # Look up tables for variables as a function of temperature and SOC
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
    P_net      = P_heat - h*cell_surface_area*(T_cell - T_ambient)
    dT_dt      = P_net/(cell_mass*Cp)
    T_current  = T_current[0] + np.dot(I,dT_dt)
    
    # Determine actual power going into the battery accounting for resistance losses
    P_loss = n_total*P_heat
    P = P_bat - np.abs(P_loss) 
    
    # Determine total energy coming from the battery in this segment
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
    V_ul   = V_oc + V_Th + (I_cell * R_0)
    
    # Determine new charge throughput (the amount of charge gone through the battery)
    Q_current = np.dot(I,abs(I_cell))
    Q_total   = Q_prior + Q_current[-1][0]/3600
    
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
    battery.internal_resistance      = R_0
    battery.state_of_charge          = SOC_new
    battery.depth_of_discharge       = DOD_new
    battery.voltage_under_load       = V_ul*n_series 
    
    return battery
    
def battery_performance_maps():
    battery_data = Data()
    T_bp = np.array([0., 20., 30., 45.])
    SOC_bp = np.array( [0. , 0.03333333, 0.06666667, 0.1 , 0.13333333, 0.16666667,
           0.2 , 0.23333333, 0.26666667, 0.3 , 0.33333333, 0.36666667,
           0.4 , 0.43333333, 0.46666667, 0.5 , 0.53333333, 0.56666667,
           0.6 , 0.63333333, 0.66666667, 0.7 , 0.73333333, 0.76666667,
           0.8 , 0.83333333, 0.86666667, 0.9 , 0.93333333, 0.96666667,
           1. ] )
     
    tV_oc = np.array([ [2.92334783,3.00653623,3.08972464,3.17291304,3.23989855,3.31010145, 3.3803913 ,
           3.44033333,3.49033333,3.52169565,3.54391304,3.58695652, 3.62095652,3.65437681,
           3.68604348,3.72430435,3.75531884,3.79102899, 3.82030435,3.84181159,3.86124638,
           3.88921739,3.91686957,3.96223188, 4.00169565,4.04117391,4.06849275,4.07573913,
           4.08571014,4.10571014, 4.161 ] , 
          [2.99293893,3.05400763,3.11507634,3.17614504, 3.23506616,3.30371247, 3.37521374,
           3.43605852,3.48697455,3.5200229 ,3.54251908,3.58374046, 3.6329313 ,3.67379644,
           3.70287532,3.73784733,3.76526463,3.79174809, 3.81922901,3.84108142,3.87212214,
           3.90738931,3.93615267,3.98113995, 4.02093893,4.04504071,4.07114758,4.07583969,
           4.08371501,4.10560814, 4.161 ] , 
          [2.84084639,2.98428484,3.1050295 ,3.19464496,3.25566531,3.309059 , 3.37185148,
           3.43473652,3.49059613,3.51955239,3.541353 ,3.58558494, 3.62641607,3.6708881 ,
           3.70814547,3.7392177 ,3.76822075,3.79592981, 3.82260427,3.84986368,3.88146592,
           3.91739674,3.94798779,3.98188403, 4.02274568,4.05623296,4.06830824,4.07468871,
           4.08175788,4.10853306, 4.153 ] ,
          [2.81925101,2.97410931,3.09861134,3.18674899,3.24142105,3.29678138, 3.35963563,
           3.42195951,3.47637247,3.51383806,3.54319838,3.59076923, 3.61940891,3.65574089,
           3.7067004 ,3.74153441,3.77023887,3.79773684, 3.82421053,3.85139271,3.88311336,
           3.91906478,3.94918219,3.98310931, 4.02401215,4.05611741,4.07036842,4.07774494,
           4.08190283,4.10867206, 4.153 ] ])
    
    tC_Th = np.array([ [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] ,
          [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] ,
          [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] , 
          [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] ])
    tR_Th = np.array([ [0.09 ,0.09 ,0.09 ,0.09 ,0.07130435 ,0.06 , 0.06 ,0.06 ,0.06 ,0.06 ,0.06 ,0.06 ,
           0.08217391, 0.07492754, 0.07 ,0.07 ,0.07 ,0.07 , 0.07 ,0.05318841,0.04144928,
           0.0573913 ,0.06884058,0.07, 0.07456522,0.075,0.075,0.05586957,0.055 ,0.04021739, 0.04 ] ,
          [0.08534351,0.07516539,0.06498728,0.05480916,0.04838931,0.04589059, 0.045 ,0.045 ,0.04195929,
           0.03937405,0.03642494,0.035 , 0.035 ,0.03848601,0.05430025,0.04534351,0.03624682,0.03115776,
           0.03, 0.03 ,0.03 ,0.03839695,0.04 ,0.04 , 0.04 ,0.03089059,0.03,0.03 ,0.02807125, 0.02505344 , 0.02],
          [0.0677823 ,0.05252289,0.045 ,0.045,0.045,0.045 , 0.04207528,0.04 ,0.03690234,0.035 ,0.0317294 ,
           0.02798576,0.027 ,0.025588 ,0.025 ,0.02129705,0.02   ,0.02 , 0.04377416,0.04190234,0.04,0.04 ,
           0.04 ,0.03121058,0.02820753,0.028,0.02055341,0.02 ,0.02,0.02 ,0.001 ] ,
          [0.06728745,0.04704453,0.04,0.04 ,0.04 ,0.04 ,  0.04 ,0.04  ,0.03267206,0.03 ,0.03 ,0.03 , 0.03,
           0.02603239,0.025 ,0.02091093,0.02,0.02 , 0.04562753,0.04133603,0.04  ,0.04  ,0.04 ,0.0308502 , 
           0.02814575,0.028 ,0.02038866,0.02 ,0.02 ,0.02,  0.001 ]]) 
    
    tR_0 = np.array([ [0.2473913 ,0.20681159,0.16623188,0.12565217,0.09753623,0.08362319, 0.08,0.07666667,0.0715942 ,0.07 ,0.0415942 ,0.05681159,
                       0.067 ,0.067 ,0.067 ,0.067 ,0.067 ,0.06537681, 0.065 ,0.065 ,0.065 ,0.065 ,0.065 ,0.065 , 0.065 ,0.065 ,0.065 ,0.065 ,0.065 ,
                       0.065 , 0.065 ] ,
                      [0.08801527,0.07274809,0.05748092,0.04221374,0.03231552,0.02722646, 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ] ,
                      [0.0677823 ,0.05252289,0.03726348,0.02733469,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 ,0.025 , 0.01311292,0.01809766,0.02 ,0.02 ,0.02430824,0.025 ,  0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.03] , 
                      [0.06546559,0.0502834 ,0.03510121,0.02663968,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 , 0.01218623,0.01,0.01 ,0.01890688,0.02451417,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.03] ])
    
    SMOOTHING = 0.1 # more is more smooth, less true to the data
    battery_data.V_oc_interp = RectBivariateSpline(T_bp, SOC_bp, tV_oc, s=SMOOTHING) # % need Deg C
    battery_data.C_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tC_Th, s=SMOOTHING) # % need Deg C
    battery_data.R_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tR_Th, s=SMOOTHING) # % need Deg C
    battery_data.R_0_interp = RectBivariateSpline(T_bp, SOC_bp, tR_0, s=SMOOTHING)   # % need Deg C
 
    return battery_data
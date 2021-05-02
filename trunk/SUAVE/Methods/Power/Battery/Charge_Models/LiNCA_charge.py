## @ingroup Methods-Power-Battery-Charge
# LiNCA_discharge.py
# 
# Created: Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units 
import numpy as np 
from scipy.integrate import  cumtrapz 

def LiNCA_charge(battery,numerics): 
    """This is a Charge model for lithium-nickel-cobalt-aluminum oxide 18650 battery
       using a thevenin equavalent circuit with parameters taken from 
       pulse tests done by NASA Glen (referece below) of a Samsung (SDI 18650-30Q).
       Cell Aging model was developed from fitting of experimental data of LiMNC 
       18650 cell.  
     
       Source: 
       Cell Charge: Chin, J. C., Schnulo, S. L., Miller, T. B., Prokopius, K., and Gray, 
       J., Battery Performance Modeling on Maxwell X-57",AIAA Scitech, San Diego, CA,
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
              battery_voltage_open_circuit                                     [Volts]
              battery_thevenin_voltage                                 [Volts]
              charge_throughput                                        [Amp-hrs]
              internal_resistance                                      [Ohms]
              battery_state_of_charge                                          [unitless]
              depth_of_discharge                                       [unitless]
              battery_voltage_under_load                                        [Volts]   
        
    """
    
    # Unpack varibles 
    I_bat                    = battery.inputs.current
    P_bat                    = battery.inputs.power_in   
    cell_mass                = battery.cell.mass   
    electrode_area           = battery.cell.electrode_area
    Cp                       = battery.cell.specific_heat_capacity 
    h                        = battery.heat_transfer_coefficient
    As_cell                  = battery.cell.surface_area 
    D_cell                   = battery.cell.diameter                     
    H_cell                   = battery.cell.height    
    T_ambient                = battery.ambient_temperature 
    T_current                = battery.pack_temperature      
    T_cell                   = battery.cell_temperature     
    E_max                    = battery.max_energy
    R_growth_factor          = battery.R_growth_factor 
    E_current                = battery.current_energy 
    Q_prior                  = battery.charge_throughput  
    battery_data             = battery.discharge_performance_map 
    heat_transfer_efficiency = battery.heat_transfer_efficiency
    I                        = numerics.time.integrate  
    D                        = numerics.time.differentiate      
    
    # ---------------------------------------------------------------------------------
    # Compute battery electrical properties 
    # ---------------------------------------------------------------------------------
    
    # Calculate the current going into one cell  
    n_series          = battery.pack_config.series  
    n_parallel        = battery.pack_config.parallel
    n_total           = n_series * n_parallel 
    Nn                = battery.module_config.normal_count            
    Np                = battery.module_config.parallel_count          
    n_total_module    = Nn*Np    
    I_cell            = -I_bat/n_parallel
    
    # State of charge of the battery
    initial_discharge_state = np.dot(I,P_bat) + E_current[0]
    SOC_old =  np.divide(initial_discharge_state,E_max)
    SOC_old[SOC_old < 0.] = 0.    
    SOC_old[SOC_old > 1.] = 1.    
    DOD_old = 1 - SOC_old          
    
    # ---------------------------------------------------------------------------------
    # Compute battery cell temperature 
    # ---------------------------------------------------------------------------------
    # Determine temperature increase         
    sigma   = 100  # Electrical conductivity  Parameter Estimation and Model Discrimination for a Lithium-Ion Cell
    n       = 1
    F       = 96485      # C/mol Faraday constant    
    delta_S = -18046*(SOC_old)**6 + 52735*(SOC_old)**5 - 57196*(SOC_old)**4 + \
               28030*(SOC_old)**3 -6023*(SOC_old)**2 +  514*(SOC_old) -27
    
    i_cell         = I_cell/electrode_area # current intensity 
    q_dot_entropy  = -(T_cell)*delta_S*i_cell/(n*F)  # temperature in Kelvin  
    q_dot_joule    = (i_cell**2)/sigma                   # eqn 5 , D. Jeon Thermal Modelling ..
    Q_heat_gen     = (q_dot_joule + q_dot_entropy)*As_cell 
    q_joule_frac   = q_dot_joule/(q_dot_joule + q_dot_entropy)
    q_entropy_frac = q_dot_entropy/(q_dot_joule + q_dot_entropy)
    
    if n_total == 1: 
        # Using lumped model   
        Q_convec       = h*As_cell*(T_cell - T_ambient) 
        P_net          = Q_heat_gen - Q_convec
        P_net          = P_net*n_total 

    else: 
        # Chapter 7 pg 437-446 of Fundamentals of heat and mass transfer : Frank P. Incropera ... Incropera, Fran
        S_T     = battery.module_config.normal_spacing          
        S_L     = battery.module_config.parallel_spacing                       
        K_air   = battery.cooling_fluid.thermal_conductivity    
        Cp_air  = battery.cooling_fluid.specific_heat_capacity  
        V_air   = battery.cooling_fluid.discharge_air_cooling_flowspeed
        rho_air = battery.cooling_fluid.density 
        nu_air  = battery.cooling_fluid.kinematic_viscosity 
        Pr_fit  = battery.cooling_fluid.prandlt_number_fit     

        S_D = np.sqrt(S_T**2+S_L**2)
        if 2*(S_D-D_cell) < (S_T-D_cell):
            V_max    = V_air*(S_T/(2*(S_D-D_cell)))
        else:
            V_max   = V_air*(S_T/(S_T-D_cell))

        T        = (T_ambient+T_current)/2   
        Re_max   = V_max*D_cell/nu_air
        Pr       = Pr_fit(T_ambient- 272.65)
        Prw      = Pr_fit(T- 272.65)  
        if all(Re_max) > 10E2: 
            C        = 0.35*((S_T/S_L)**0.2) 
            m        = 0.6 
        else:
            C = 0.51
            m = 0.5 
        Nu       = C*(Re_max**m)*(Pr**0.36)*((Pr/Prw)**0.25)           
        h        = Nu*K_air/D_cell
        Tw_Ti    = (T - T_ambient)
        Tw_To    = Tw_Ti * np.exp((-np.pi*D_cell*n_total_module*h)/(rho_air*V_air*Nn*S_T*Cp_air))
        dT_lm    = (Tw_Ti - Tw_To)/np.log(Tw_Ti/Tw_To)
        Q_convec = heat_transfer_efficiency*h*np.pi*D_cell*H_cell*n_total_module*dT_lm  
        P_net    = Q_heat_gen*n_total_module - Q_convec 

    dT_dt     = P_net/(cell_mass*n_total_module*Cp)
    T_current = T_current[0] + np.dot(I,dT_dt)  

    # Power going into the battery accounting for resistance losses
    P_loss = n_total*Q_heat_gen
    P      = P_bat - np.abs(P_loss)      
    
    # Look up tables for variables as a function of temperature and SOC 
    V_oc = np.zeros_like(I_cell)
    R_Th = np.zeros_like(I_cell)  
    C_Th = np.zeros_like(I_cell)  
    R_0  = np.zeros_like(I_cell) 
    for i in range(len(SOC_old)): 
        T_cell_Celcius = T_cell[i]- 272.65
        
        # Open Circuit Voltage
        V_oc[i] = battery_data.V_oc_interp(T_cell_Celcius, SOC_old[i])[0]
        
        # Thevenin Capacitance 
        C_Th[i] = battery_data.C_Th_interp(T_cell_Celcius, SOC_old[i])[0]
        
        # Thevenin Resistance 
        R_Th[i] = battery_data.R_Th_interp(T_cell_Celcius, SOC_old[i])[0]
        
        # Li-ion battery interal resistance
        R_0[i]  = battery_data.R_0_interp(T_cell_Celcius, SOC_old[i])[0] 
        
    
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
    
    # Add this to the current state
    if np.isnan(E_bat).any():
        E_bat=np.ones_like(E_bat)*np.max(E_bat)
        if np.isnan(E_bat.any()): #all nans; handle this instance
            E_bat = np.zeros_like(E_bat)
            
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
    Q_segment  = np.atleast_2d(np.hstack(( np.zeros_like(Q_prior[0]) , cumtrapz(I_cell[:,0], x = numerics.time.control_points[:,0])/Units.hr ))).T  
  
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
    battery.cumulative_cell_charge_throughput   = Q_total 
    battery.cell_charge_throughput              = Q_segment
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
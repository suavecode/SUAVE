## @ingroup Methods-Power-Battery 
# compute_net_generated_battery_heat.py
# 
# Created: Sep 2021, M. Clarke 
import numpy as np 

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery 
def compute_net_generated_battery_heat(n_total,battery,Q_heat_gen,numerics):
    '''Computes the net heat generated in a battery during cycling. 

    Assumptions:
    1) Battery pack cell heat transfer can be modelled as a cooling columns in a cross-flow
    2) Isothermal battery cell - the temperature at the center of the cell is the same at 
    the surface of the cell

    Source: 
    Heat Transfer Model:
    Pakowski, Zdzisław. "Fundamentals of Heat and Mass Transfer, Frank P Incropera,
    David P DeWitt, Theodore L Bergman, Adrienne S Lavine, J. Wiley & Sons, Hoboken
    NJ (2007), 997 pp." (2007): 1683-1684.,  Chapter 7 pg 437-446 

    Inputs:  
        battery. 
              h                         (heat transfer coefficient)  [W/(m^2*K)] 
              As_cell                   (battery cell surface area)  [meters^2]
              H_cell                    (battery cell height)        [meters]
              T_ambient                 (ambient temperature)        [Kelvin]
              T_current                 (pack temperature)           [Kelvin]
              T_cell                    (battery cell temperature)   [Kelvin] 
              heat_transfer_efficiency                               [unitless]
      
      Outputs:
        battery. 
             net_power                                               [Watts] 
 

    Properties Used:
    None 
    '''

    h                        = battery.convective_heat_transfer_coefficient
    As_cell                  = battery.cell.surface_area 
    D_cell                   = battery.cell.diameter                     
    H_cell                   = battery.cell.height     
    T_ambient                = battery.ambient_temperature 
    T_current                = battery.pack_temperature      
    T_cell                   = battery.cell_temperature       
    cell_mass                = battery.cell.mass    
    Cp                       = battery.cell.specific_heat_capacity       
    I                        = numerics.time.integrate      
    heat_transfer_efficiency = battery.heat_transfer_efficiency   

    # Calculate the current going into one cell   
    n_total           = battery.pack_config.total
    Nn                = battery.module_config.normal_count            
    Np                = battery.module_config.parallel_count          
    n_total_module    = Nn*Np    
    
    if n_total == 1: 
        # Using lumped model   
        Q_convec       = h*As_cell*(T_cell - T_ambient)
        P_net          = Q_heat_gen - Q_convec

    else: 
        # Chapter 7 pg 437-446 of Fundamentals of heat and mass transfer 
        S_T             = battery.module_config.normal_spacing          
        S_L             = battery.module_config.parallel_spacing   
        coolant         = battery.cooling_fluid
        K_coolant       = coolant.thermal_conductivity 
        Cp_coolant      = coolant.specific_heat_capacity  
        V_coolant       = coolant.cooling_flowspeed
        rho_coolant     = coolant.density 
        nu_coolant      = coolant.kinematic_viscosity     
        Pr_coolant      = coolant.prandtl_number  

        S_D = np.sqrt(S_T**2+S_L**2)
        if 2*(S_D-D_cell) < (S_T-D_cell):
            V_max = V_coolant*(S_T/(2*(S_D-D_cell)))
        else:
            V_max = V_coolant*(S_T/(S_T-D_cell))

        T        = (T_ambient+T_current)/2   
        Re_max   = V_max*D_cell/nu_coolant   
        if all(Re_max) > 10E2: 
            C = 0.35*((S_T/S_L)**0.2) 
            m = 0.6 
        else:
            C = 0.51
            m = 0.5  

        Pr_w_coolant          = coolant.compute_prandtl_number(T)            
        Nu                    = C*(Re_max**m)*(Pr_coolant**0.36)*((Pr_coolant/Pr_w_coolant)**0.25)           
        h                     = Nu*K_coolant/D_cell
        Tw_Ti                 = (T - T_ambient)
        Tw_To                 = Tw_Ti * np.exp((-np.pi*D_cell*n_total_module*h)/(rho_coolant*V_coolant*Nn*S_T*Cp_coolant))
        dT_lm                 = (Tw_Ti - Tw_To)/np.log(Tw_Ti/Tw_To)
        Q_convec              = heat_transfer_efficiency*h*np.pi*D_cell*H_cell*n_total_module*dT_lm 
        Q_convec[Tw_Ti == 0.] = 0.
        P_net                 = Q_heat_gen*n_total_module - Q_convec 

    dT_dt     = P_net/(cell_mass*n_total_module*Cp)
    T_current = T_current[0] + np.dot(I,dT_dt)  
    T_current[T_ambient>T_current] = T_ambient[T_ambient>T_current]
    
    return  T_current
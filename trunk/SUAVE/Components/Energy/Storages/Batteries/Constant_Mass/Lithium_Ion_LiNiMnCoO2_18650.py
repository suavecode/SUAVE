## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion_LiNiMnCoO2_18650.py
# 
# Created:  Feb 2020, M. Clarke
# Modified: Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
import numpy as np
import os
from SUAVE.Core                                                         import Units , Data 
from scipy.interpolate                                                  import RegularGridInterpolator 
from .Lithium_Ion                                                       import Lithium_Ion 
from scipy.integrate import  cumtrapz , odeint 
from SUAVE.Methods.Power.Battery.Cell_Cycle_Models.LiNiMnCoO2_cell_cycle_model import compute_NMC_cell_state_variables

## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion_LiNiMnCoO2_18650(Lithium_Ion):
    """ Specifies discharge/specific energy characteristics specific 
        18650 lithium-nickel-manganese-cobalt-oxide battery cells     
        
        Assumptions:
        Convective Thermal Conductivity Coefficient corresponds to forced
        air cooling in 35 m/s air 
        
        Source:
        Automotive Industrial Systems Company of Panasonic Group, Technical Information of 
        NCR18650G, URL https://www.imrbatteries.com/content/panasonic_ncr18650g.pdf
        
        convective  heat transfer coefficient, h 
        Jeon, Dong Hyup, and Seung Man Baek. "Thermal modeling of cylindrical 
        lithium ion battery during discharge cycle." Energy Conversion and Management
        52.8-9 (2011): 2973-2981.
        
        thermal conductivity, k 
        Yang, Shuting, et al. "A Review of Lithium-Ion Battery Thermal Management 
        System Strategies and the Evaluate Criteria." Int. J. Electrochem. Sci 14
        (2019): 6077-6107.
        
        specific heat capacity, Cp
        (axial and radial)
        Yang, Shuting, et al. "A Review of Lithium-Ion Battery Thermal Management 
        System Strategies and the Evaluate Criteria." Int. J. Electrochem. Sci 14
        (2019): 6077-6107.
        
        # Electrode Area
        Muenzel, Valentin, et al. "A comparative testing study of commercial
        18650-format lithium-ion battery cells." Journal of The Electrochemical
        Society 162.8 (2015): A1592.
        
        Inputs:
        None
        
        Outputs:
        None
        
        Properties Used:
        N/A
    """       
    
    def __defaults__(self):    
        self.tag                              = 'Lithium_Ion_LiNiMnCoO2_Cell' 
                                              
        self.cell.diameter                    = 0.0184                                                   # [m]
        self.cell.height                      = 0.0652                                                   # [m]
        self.cell.mass                        = 0.048 * Units.kg                                         # [kg]
        self.cell.surface_area                = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)   # [m^2]
        self.cell.volume                      = np.pi*(0.5*self.cell.diameter)**2*self.cell.height       # [m^3] 
        self.cell.density                     = self.cell.mass/self.cell.volume                          # [kg/m^3]  
        self.cell.electrode_area              = 0.0342                                                   # [m^2] 
                                                                                               
        self.cell.max_voltage                 = 4.2                                                      # [V]
        self.cell.nominal_capacity            = 3.55                                                     # [Amp-Hrs]
        self.cell.nominal_voltage             = 3.6                                                      # [V] 
        self.cell.charging_voltage            = self.cell.nominal_voltage                                # [V] 
        
        self.watt_hour_rating                 = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
        self.specific_energy                  = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
        self.specific_power                   = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
        self.resistance                       = 0.025                                                    # [Ohms]
                                                                                                         
        self.specific_heat_capacity           = 1108                                                     # [J/kgK]  
        self.cell.specific_heat_capacity      = 1108                                                     # [J/kgK]    
        self.cell.radial_thermal_conductivity = 0.4                                                      # [J/kgK]  
        self.cell.axial_thermal_conductivity  = 32.2                                                     # [J/kgK] # estimated  
                                              
        battery_raw_data                      = load_battery_results()                                                   
        self.discharge_performance_map        = create_discharge_performance_map(battery_raw_data)  
        
        return  
    
    def energy_calc(self,numerics,battery_discharge_flag = True ): 
        '''This is a electric cycle model for 18650 lithium-nickel-manganese-cobalt-oxide
           battery cells. The model uses experimental data performed
           by the Automotive Industrial Systems Company of Panasonic Group 
              
           Source:  
           Internal Resistance Model:
           Zou, Y., Hu, X., Ma, H., and Li, S. E., "Combined State of Charge and State of
           Health estimation over lithium-ion battery cellcycle lifespan for electric 
           vehicles,"Journal of Power Sources, Vol. 273, 2015, pp. 793-803.
           doi:10.1016/j.jpowsour.2014.09.146,URLhttp://dx.doi.org/10.1016/j.jpowsour.2014.09.146. 
           
           Assumtions:
           1) All battery modules exhibit the same themal behaviour. 
           
           Inputs:
             battery.
                   I_bat             (max_energy)                          [Joules]
                   cell_mass         (battery cell mass)                   [kilograms]
                   Cp                (battery cell specific heat capacity) [J/(K kg)]
                   h                 (heat transfer coefficient)           [W/(m^2*K)]
                   t                 (battery age in days)                 [days]
                   As_cell           (battery cell surface area)           [meters^2]
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
                  charge_throughput                                        [Amp-hrs]
                  internal_resistance                                      [Ohms]
                  battery_state_of_charge                                  [unitless]
                  depth_of_discharge                                       [unitless]
                  battery_voltage_under_load                               [Volts]
           
        '''
        # Unpack varibles 
        battery                  = self
        I_bat                    = battery.inputs.current
        P_bat                    = battery.inputs.power_in   
        cell_mass                = battery.cell.mass   
        electrode_area           = battery.cell.electrode_area
        Cp                       = battery.cell.specific_heat_capacity 
        h                        = battery.convective_heat_transfer_coefficient
        As_cell                  = battery.cell.surface_area 
        D_cell                   = battery.cell.diameter                     
        H_cell                   = battery.cell.height    
        P_ambient                = battery.ambient_pressure
        T_ambient                = battery.ambient_temperature 
        V_th0                    = battery.initial_thevenin_voltage 
        T_current                = battery.pack_temperature      
        T_cell                   = battery.cell_temperature     
        E_max                    = battery.max_energy
        R_growth_factor          = battery.R_growth_factor 
        E_current                = battery.current_energy 
        Q_prior                  = battery.charge_throughput  
        battery_data             = battery.discharge_performance_map 
        heat_transfer_efficiency = battery.heat_transfer_efficiency
        I                        = numerics.time.integrate  
          
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

        if battery_discharge_flag:
            I_cell = I_bat/n_parallel
        else: 
            I_cell = -I_bat/n_parallel 
        
        # State of charge of the battery
        initial_discharge_state = np.dot(I,P_bat) + E_current[0]
        SOC_old =  np.divide(initial_discharge_state,E_max) 
          
        # Make sure things do not break by limiting current, temperature and current 
        SOC_old[SOC_old < 0.] = 0.  
        SOC_old[SOC_old > 1.] = 1.   
        
        T_cell[T_cell<272.65]  = 272.65
        T_cell[T_cell>322.65]  = 322.65
        
        # ---------------------------------------------------------------------------------
        # Compute battery cell temperature 
        # ---------------------------------------------------------------------------------
        # Determine temperature increase         
        sigma   = 139 # Electrical conductivity
        n       = 1
        F       = 96485 # C/mol Faraday constant    
        delta_S = -496.66*(SOC_old)**6 +  1729.4*(SOC_old)**5 + -2278 *(SOC_old)**4 +  1382.2 *(SOC_old)**3 + \
                  -380.47*(SOC_old)**2 + 46.508*(SOC_old) + -10.692  # eqn 10 and , D. Jeon Thermal Modelling ..
        
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
            S_T             = battery.module_config.normal_spacing          
            S_L             = battery.module_config.parallel_spacing                      
            coolant         = SUAVE.Attributes.Gases.Air()
            coolant.K       = battery.cooling_fluid.thermal_conductivity    
            coolant.Cp      = battery.cooling_fluid.specific_heat_capacity  
            coolant.V       = battery.cooling_fluid.discharge_air_cooling_flowspeed
            coolant.rho     = battery.cooling_fluid.density 
            coolant.nu      = battery.cooling_fluid.kinematic_viscosity     
        
            S_D = np.sqrt(S_T**2+S_L**2)
            if 2*(S_D-D_cell) < (S_T-D_cell):
                V_max = coolant.V*(S_T/(2*(S_D-D_cell)))
            else:
                V_max = coolant.V*(S_T/(S_T-D_cell))
        
            T      = (T_ambient+T_current)/2   
            Re_max = V_max*D_cell/coolant.nu   
            if all(Re_max) > 10E2: 
                C = 0.35*((S_T/S_L)**0.2) 
                m = 0.6 
            else:
                C = 0.51
                m = 0.5 
                
            coolant.Pr      = coolant.compute_prandlt_number(T_ambient,P_ambient)
            coolant.Pr_wall = coolant.compute_prandlt_number(T,P_ambient)
            Nu              = C*(Re_max**m)*(coolant.Pr**0.36)*((coolant.Pr/coolant.Pr_wall)**0.25)           
            h               = Nu*coolant.K/D_cell
            Tw_Ti           = (T - T_ambient)
            Tw_To           = Tw_Ti * np.exp((-np.pi*D_cell*n_total_module*h)/(coolant.rho*coolant.V*Nn*S_T*coolant.Cp))
            dT_lm           = (Tw_Ti - Tw_To)/np.log(Tw_Ti/Tw_To)
            Q_convec        = heat_transfer_efficiency*h*np.pi*D_cell*H_cell*n_total_module*dT_lm 
            P_net           = Q_heat_gen*n_total_module - Q_convec  
        
        dT_dt     = P_net/(cell_mass*n_total_module*Cp)
        T_current = T_current[0] + np.dot(I,dT_dt)  
        
        # Power going into the battery accounting for resistance losses
        P_loss = n_total*Q_heat_gen
        P = P_bat - np.abs(P_loss)      
        
        # Compute State Variables
        V_ul  = compute_NMC_cell_state_variables(battery_data,SOC_old,T_cell,I_cell)  
            
        # Thevenin Time Constnat 
        tau_Th  =   2.151* np.exp(2.132 *SOC_old) + 27.2 
        
        # Thevenin Resistance 
        R_Th    =  -1.212* np.exp(-0.03383*SOC_old) + 1.258
         
        # Thevenin Capacitance 
        C_Th     = tau_Th/R_Th
        
        # Li-ion battery interal resistance
        R_0      =  0.01483*(SOC_old**2) - 0.02518*SOC_old + 0.1036 
        
        # Update battery internal and thevenin resistance with aging factor
        R_0_aged = R_0 * R_growth_factor
         
        # Compute thevening equivalent voltage   
        V_th0  = V_th0/n_series
        V_Th   = compute_thevenin_voltage(V_th0,I_cell,C_Th ,R_Th,numerics.time.control_points[:,0])
        
        # Voltage under load: 
        V_oc      = V_ul + V_Th + (I_cell * R_0_aged) 
        
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
        
        # Determine new charge throughput (the amount of charge gone through the battery)
        Q_total    = np.atleast_2d(np.hstack(( Q_prior[0] , Q_prior[0] + cumtrapz(I_cell[:,0], x = numerics.time.control_points[:,0])/Units.hr ))).T   
        
        # If SOC is negative, voltage under load goes to zero 
        V_ul[SOC_new < 0.] = 0.
            
        # Pack outputs
        battery.current_energy                     = E_current
        battery.cell_temperature                   = T_current 
        battery.pack_temperature                   = T_current 
        battery.cell_joule_heat_fraction           = q_joule_frac
        battery.cell_entropy_heat_fraction         = q_entropy_frac
        battery.resistive_losses                   = P_loss
        battery.load_power                         = V_ul*n_series*I_bat
        battery.current                            = I_bat
        battery.voltage_open_circuit               = V_oc*n_series
        battery.cell_voltage_open_circuit          = V_oc
        battery.cell_current                       = I_cell
        battery.thevenin_voltage                   = V_Th*n_series
        battery.charge_throughput                  = Q_total*n_parallel  
        battery.cell_charge_throughput             = Q_total
        battery.heat_energy_generated              = Q_heat_gen*n_total_module    
        battery.internal_resistance                = R_0*n_series
        battery.state_of_charge                    = SOC_new
        battery.depth_of_discharge                 = DOD_new
        battery.voltage_under_load                 = V_ul*n_series 
        battery.cell_voltage_under_load            = V_ul
        
        return battery 
    
    def append_battery_unknowns(self,segment): 
        """ Appends unknowns specific to NMC cells which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment.state.unknowns.battery_cell_temperature   [Kelvin]
            segment.state.unknowns.battery_state_of_charge    [unitless]
            segment.state.unknowns.battery_current            [Amperes]
    
            Outputs: 
            segment.state.conditions.propulsion.battery_cell_temperature  [Kelvin]  
            segment.state.conditions.propulsion.battery_state_of_charge   [unitless]
            segment.state.conditions.propulsion.battery_current           [Amperes]
    
            Properties Used:
            N/A
        """             
        
        segment.state.conditions.propulsion.battery_cell_temperature    = segment.state.unknowns.battery_cell_temperature 
        segment.state.conditions.propulsion.battery_state_of_charge     = segment.state.unknowns.battery_state_of_charge
        segment.state.conditions.propulsion.battery_current             = segment.state.unknowns.battery_current    
        
        return     
    

    def append_battery_residuals(self,segment,network): 
        """ This packs the residuals specific to NMC cells to be sent to the mission solver.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            segment.state.conditions.propulsion:
                battery_state_of_charge      [unitless] 
                battery_cell_temperature     [Kelvin]        
                battery_current              [Amperes]
            segment.state.unknowns.
                battery_state_of_charge      [unitless]
                battery_cell_temperature     [Kelvin]  
                battery_current              [Amperes]
            Outputs:
            None
    
            Properties Used:
            None
        """      
        
        SOC_actual   = segment.state.conditions.propulsion.battery_state_of_charge
        SOC_predict  = segment.state.unknowns.battery_state_of_charge 
    
        Temp_actual  = segment.state.conditions.propulsion.battery_cell_temperature 
        Temp_predict = segment.state.unknowns.battery_cell_temperature   
    
        i_actual     = segment.state.conditions.propulsion.battery_current
        i_predict    = segment.state.unknowns.battery_current      
    
        # Return the residuals  
        segment.state.residuals.network.SOC         =  SOC_predict[:,0]  - SOC_actual[:,0]  
        segment.state.residuals.network.temperature =  Temp_predict[:,0] - Temp_actual[:,0]
        segment.state.residuals.network.current     =  i_predict[:,0]    - i_actual[:,0]  
        
        return  num_residuals
    
    def append_battery_unknowns_and_residuals_to_segment(self,segment,initial_voltage,
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current,initial_battery_cell_thevenin_voltage): 
        """ This function sets up the information that the mission needs to run a mission segment using this network
    
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
        ones_row = segment.state.ones_row  
         
        parallel                                           = self.pack_config.parallel            
        segment.state.unknowns.battery_state_of_charge     = initial_battery_state_of_charge   * ones_row(1)  
        segment.state.unknowns.battery_cell_temperature    = initial_battery_cell_temperature  * ones_row(1) 
        segment.state.unknowns.battery_current             = initial_battery_cell_current*parallel * ones_row(1)  
        
        return  

def compute_thevenin_voltage(V_th0,I,C_Th, R_Th,t):
    """ Computes the thevenin voltage of an NMC cell using SciPy ODE solver 
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:   
        V_th0 - initial thevenin voltage [Volts]
        I     - cell current             [Amperes]
        C_th  - thevenin capacitance     [Coulombs]
        R_th  - thevenin resistnace      [Ohms]
        t     - discretized time         [seconds]

    Outputs:  
        V_th  - thevenin voltage         [Volts]
        
    Properties Used:
    N/A 
    """     
    n    = len(t)
    V_th = np.zeros(n)
    
    # Initial conditition
    V_th[0] = V_th0 
    
    for i in range(1,n): 
        z = odeint(model, V_th0, t, args=(I[i][0],C_Th[i][0], R_Th[i][0])) 
        z0 = z[1] 
        V_th[i] = z0[0] 
        
    return np.atleast_2d(V_th).T
     
def model(z,t,I,C_Th, R_Th):
    """ Computes the derivative of the thevenin voltage for the ODE solver
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:   
        z        - ODE function                                [Unitless]
        I        - cell current                                [Amperes]
        C_th     - thevenin capacitance                        [Coulombs]
        R_th     - thevenin resistnace                         [Ohms]
        t        - discretized time                            [seconds]

    Outputs:  
        dVth_dt  - derivative of thevenin voltage w.r.t time   [Volts/time]
        
    Properties Used:
    N/A 
    """    
    V_th    = z[0]
    dVth_dt = I/C_Th - (V_th/(R_Th*C_Th))
    return [dVth_dt] 

def create_discharge_performance_map(battery_raw_data):
    """ Create discharge and charge response surface for 
        LiNiMnCoO2 battery cells 
        
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
    
    # Process raw data 
    processed_data = process_raw_data(battery_raw_data)
    
    # Create performance maps 
    battery_data = create_response_surface(processed_data) 
    
    return battery_data

def create_response_surface(processed_data):
    
    battery_map             = Data() 
    amps                    = np.linspace(0, 8, 5)
    temp                    = np.linspace(0, 50, 6) +  272.65
    SOC                     = np.linspace(0, 1, 15) 
    battery_map.Voltage     = RegularGridInterpolator((amps, temp, SOC), processed_data.Voltage)
    battery_map.Temperature = RegularGridInterpolator((amps, temp, SOC), processed_data.Temperature)
     
    return battery_map 

def process_raw_data(raw_data):
    """ Takes raw data and formats voltage as a function of SOC, current and temperature
        
        Source 
        N/A
        
        Assumptions:
        N/A
        
        Inputs:
        raw_Data     
            
        Outputs: 
        procesed_data 

        Properties Used:
        N/A
                                
    """
    processed_data = Data()
     
    processed_data.Voltage        = np.zeros((5,6,15,2)) # current , operating temperature , SOC vs voltage      
    processed_data.Temperature    = np.zeros((5,6,15,2)) # current , operating temperature , SOC  vs temperature 
    
    # Reshape  Data          
    raw_data.Voltage 
    for i, Amps in enumerate(raw_data.Voltage):
        for j , Deg in enumerate(Amps):
            min_x    = 0 
            max_x    = max(Deg[:,0])
            x        = np.linspace(min_x,max_x,15)
            y        = np.interp(x,Deg[:,0],Deg[:,1])
            vec      = np.zeros((15,2))
            vec[:,0] = x/max_x
            vec[:,1] = y
            processed_data.Voltage[i,j,:,:]= vec   
            
    for i, Amps in enumerate(raw_data.Temperature):
        for j , Deg in enumerate(Amps):
            min_x    = 0   
            max_x    = max(Deg[:,0])
            x        = np.linspace(min_x,max_x,15)
            y        = np.interp(x,Deg[:,0],Deg[:,1])
            vec      = np.zeros((15,2))
            vec[:,0] = x/max_x
            vec[:,1] = y
            processed_data.Temperature[i,j,:,:]= vec     
    
    return  processed_data  

def load_battery_results():
    ospath    = os.path.abspath(__file__)
    separator = os.path.sep
    rel_path  = os.path.dirname(ospath) + separator     
    return SUAVE.Input_Output.SUAVE.load(rel_path+ 'NMC_Raw_Data.res')

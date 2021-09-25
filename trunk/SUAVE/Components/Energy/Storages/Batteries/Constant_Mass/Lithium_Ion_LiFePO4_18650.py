## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion_LiFePO4_18650.py
# 
# Created:  Feb 2020, M. Clarke
# Modified: Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports 
from SUAVE.Core import Units  
from .Lithium_Ion import Lithium_Ion 

# package imports 
import numpy as np 
from scipy.integrate import  cumtrapz

## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion_LiFePO4_18650(Lithium_Ion):
    """ Specifies discharge/specific energy characteristics specific 
        18650 lithium-iron-phosphate-oxide battery cells.     
        
        Assumptions:
        Convective Thermal Conductivity Coefficient corresponds to forced
        air cooling in 35 m/s air 
        
        Source:
        # Cell Information 
        Saw, L. H., Yonghuang Ye, and A. A. O. Tay. "Electrochemical–thermal analysis of 
        18650 Lithium Iron Phosphate cell." Energy Conversion and Management 75 (2013): 
        162-174.
        
        # Electrode Area
        Muenzel, Valentin, et al. "A comparative testing study of commercial
        18650-format lithium-ion battery cells." Journal of The Electrochemical
        Society 162.8 (2015): A1592.
        
        # Cell Thermal Conductivities 
        (radial)
        Murashko, Kirill A., Juha Pyrhönen, and Jorma Jokiniemi. "Determination of the 
        through-plane thermal conductivity and specific heat capacity of a Li-ion cylindrical 
        cell." International Journal of Heat and Mass Transfer 162 (2020): 120330.
        
        (axial)
        Saw, L. H., Yonghuang Ye, and A. A. O. Tay. "Electrochemical–thermal analysis of 
        18650 Lithium Iron Phosphate cell." Energy Conversion and Management 75 (2013): 
        162-174.
        
        Inputs:
        None
        
        Outputs:
        None
        
        Properties Used:
        N/A
        """ 
    def __defaults__(self):
        self.tag                              = 'Lithium_Ion_LiFePO4_Cell' 
         
        self.cell.diameter                    = 0.0185                                                   # [m]
        self.cell.height                      = 0.0653                                                   # [m]
        self.cell.mass                        = 0.03  * Units.kg                                         # [kg]
        self.cell.surface_area                = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)  # [m^2]
        self.cell.volume                      = np.pi*(0.5*self.cell.diameter)**2*self.cell.height       # [m^3] 
        self.cell.density                     = self.cell.mass/self.cell.volume                          # [kg/m^3]
        self.cell.electrode_area              = 0.0342                                                   # [m^2]  # estimated 
                                                        
        self.cell.max_voltage                 = 3.6                                                      # [V]
        self.cell.nominal_capacity            = 1.5                                                      # [Amp-Hrs]
        self.cell.nominal_voltage             = 3.6                                                      # [V]
        self.cell.charging_voltage            = self.cell.nominal_voltage                                # [V]  
         
        self.watt_hour_rating                 = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
        self.specific_energy                  = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
        self.specific_power                   = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
        self.ragone.const_1                   = 88.818  * Units.kW/Units.kg
        self.ragone.const_2                   = -.01533 / (Units.Wh/Units.kg)
        self.ragone.lower_bound               = 60.     * Units.Wh/Units.kg
        self.ragone.upper_bound               = 225.    * Units.Wh/Units.kg         
        self.resistance                       = 0.022                                                    # [Ohms]
                                                        
        self.specific_heat_capacity           = 1115                                                     # [J/kgK] 
        self.cell.specific_heat_capacity      = 1115                                                     # [J/kgK] 
        self.cell.radial_thermal_conductivity = 0.475                                                    # [J/kgK]  
        self.cell.axial_thermal_conductivity  = 37.6                                                     # [J/kgK]   
        
        return   

    def energy_calc(self,numerics,battery_discharge_flag= True): 
        """This is an electric cycle model for 18650 lithium-iron_phosphate battery cells. It
           models losses based on an empirical correlation Based on method taken 
           from Datta and Johnson.
           
           Assumptions: 
           1) Constant Peukart coefficient
           2) All battery modules exhibit the same themal behaviour.
           
           Source:
           Internal Resistance:
           Nikolian, Alexandros, et al. "Complete cell-level lithium-ion electrical ECM model 
           for different chemistries (NMC, LFP, LTO) and temperatures (− 5° C to 45° C)–
           Optimized modelling techniques." International Journal of Electrical Power &
           Energy Systems 98 (2018): 133-146.
          
           Voltage:
           Chen, M. and Rincon-Mora, G. A., "Accurate Electrical
           Battery Model Capable of Predicting Runtime and I - V Performance" IEEE
           Transactions on Energy Conversion, Vol. 21, No. 2, June 2006, pp. 504-511
           
           Inputs:
             battery. 
                   I_bat             (max_energy)                          [Joules]
                   cell_mass         (battery cell mass)                   [kilograms]
                   Cp                (battery cell specific heat capacity) [J/(K kg)] 
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
            
        """ 
         
        # Unpack varibles 
        battery           = self
        I_bat             = battery.inputs.current
        P_bat             = battery.inputs.power_in   
        V_max             = battery.max_voltage
        bat_mass          = battery.mass_properties.mass                
        bat_Cp            = battery.specific_heat_capacity    
        T_current         = battery.pack_temperature   
        E_max             = battery.max_energy
        E_current         = battery.current_energy 
        Q_prior           = battery.charge_throughput 
        R_growth_factor   = battery.R_growth_factor
        E_growth_factor   = battery.E_growth_factor 
        I                 = numerics.time.integrate  

        if not battery_discharge_flag:   
            I_bat = -I_bat  
        # ---------------------------------------------------------------------------------
        # Compute battery electrical properties 
        # --------------------------------------------------------------------------------- 
        n_parallel        = battery.pack_config.parallel  
         
        # Update battery capacitance (energy) with aging factor
        E_max = E_max*E_growth_factor
        
        # Compute state of charge and depth of discarge of the battery
        initial_discharge_state = np.dot(I,P_bat) + E_current[0]
        SOC_old                 = np.divide(initial_discharge_state,E_max)  
        
        # Compute internal resistance
        R_bat = -0.0169*(SOC_old**4) + 0.0418*(SOC_old**3) - 0.0273*(SOC_old**2) + 0.0069*(SOC_old) + 0.0043
        R_0   = R_bat*R_growth_factor 
        
        # Compute Heat power generated by all cells
        Q_heat_gen = (I_bat**2.)*R_0
        
        # Determine actual power going into the battery accounting for resistance losses
        P = P_bat - np.abs(Q_heat_gen) 
        
        # Compute temperature rise of battery
        dT_dt     = Q_heat_gen /(bat_mass*bat_Cp)
        T_current = T_current[0] + np.dot(I,dT_dt)
        
        E_bat = np.dot(I,P)
        E_bat = np.reshape(E_bat,np.shape(E_current)) #make sure it's consistent
        
        # Add this to the current state
        if np.isnan(E_bat).any():
            E_bat=np.ones_like(E_bat)*np.max(E_bat)
            if np.isnan(E_bat.any()): #all nans; handle this instance
                E_bat=np.zeros_like(E_bat)
                
        E_current = E_bat + E_current[0]
        
        SOC_new = np.divide(E_current,E_max)
        SOC_new[SOC_new>1] = 1.
        SOC_new[SOC_new<0] = 0. 
        DOD_new = 1 - SOC_new
          
        # Determine new charge throughput (the amount of charge gone through the battery)
        Q_total    = np.atleast_2d(np.hstack(( Q_prior[0] , Q_prior[0] + cumtrapz(abs(I_bat)[:,0], x   = numerics.time.control_points[:,0])/Units.hr ))).T      
                
        # A voltage model from Chen, M. and Rincon-Mora, G. A., "Accurate Electrical Battery Model Capable of Predicting
        # Runtime and I - V Performance" IEEE Transactions on Energy Conversion, Vol. 21, No. 2, June 2006, pp. 504-511
        V_normalized  = (-1.031*np.exp(-35.*SOC_new) + 3.685 + 0.2156*SOC_new - 0.1178*(SOC_new**2.) + 0.3201*(SOC_new**3.))/4.1
        V_oc = V_normalized * V_max
        V_oc[ V_oc > V_max] = V_max
        
        # Voltage under load:
        if battery_discharge_flag:
            V_ul   = V_oc - I_bat*R_0
        else: 
            V_ul   = V_oc + I_bat*R_0 
             
        # Pack outputs
        battery.current_energy                     = E_current
        battery.resistive_losses                   = Q_heat_gen
        battery.cell_temperature                   = T_current 
        battery.pack_temperature                   = T_current 
        battery.load_power                         = V_ul*I_bat
        battery.state_of_charge                    = SOC_new 
        battery.depth_of_discharge                 = DOD_new
        battery.charge_throughput                  = Q_total 
        battery.cell_charge_throughput             = Q_total/n_parallel  
        battery.voltage_open_circuit               = V_oc
        battery.voltage_under_load                 = V_ul
        battery.current                            = I_bat 
        battery.pack_temperature                   = T_current 
        battery.cell_joule_heat_fraction           = np.zeros_like(V_ul)
        battery.cell_entropy_heat_fraction         = np.zeros_like(V_ul)  
        battery.cell_voltage_open_circuit          = np.zeros_like(V_ul)
        battery.cell_current                       = np.zeros_like(V_ul)
        battery.thevenin_voltage                   = np.zeros_like(V_ul)
        battery.heat_energy_generated              = Q_heat_gen 
        battery.internal_resistance                = R_0 
        battery.cell_voltage_under_load            = V_ul 
        
        return      
    
    def append_battery_unknowns(self,segment): 
        """ Appends unknowns specific to LFP cells which are unpacked from the mission solver and send to the network.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            state.unknowns.battery_voltage_under_load               [volts]
    
            Outputs: 
            state.conditions.propulsion.battery_voltage_under_load  [volts]
    
            Properties Used:
            N/A
        """             
        
        segment.state.conditions.propulsion.battery_voltage_under_load  = segment.state.unknowns.battery_voltage_under_load
        
        return 
    
    def append_battery_residuals(self,segment,network): 
        """ Packs the residuals specific to LFP cells to be sent to the mission solver.
    
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
            network.voltage                           [volts]
        """     
        v_actual  = segment.state.conditions.propulsion.battery_voltage_under_load
        v_predict = segment.state.unknowns.battery_voltage_under_load
        v_max     = network.voltage
        
        # Return the residuals
        segment.state.residuals.network.voltage = (v_predict[:,0] - v_actual[:,0])/v_max
        
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
        
        ones_row = segment.state.ones_row 
        if initial_voltage==None:
            initial_voltage = self.max_voltage 
        segment.state.unknowns.battery_voltage_under_load  = initial_voltage * ones_row(1) 
        
        return  
    
    def compute_voltage(self,state):
        """ Computes the voltage of a single LFP cell or a battery pack of LFP cells   
    
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
        battery                          = self 
        
        # Set battery properties
        battery.battery_thevenin_voltage = 0     
        battery.temperature              = state.conditions.propulsion.battery_pack_temperature 
        
        # Voltage under load
        V_ul                             = state.unknowns.battery_voltage_under_load
        return V_ul  
    
    def update_battery_age(self,segment):    
        return  
 
     
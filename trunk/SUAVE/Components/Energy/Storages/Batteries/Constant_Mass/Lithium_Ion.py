## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion.py
# 
# Created:  Nov 2014, M. Vegh
# Modified: Feb 2016, T. MacDonald
#           Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports 
from SUAVE.Core import Units, Data
from SUAVE.Attributes.Gases import Air
from SUAVE.Components.Energy.Storages.Batteries  import Battery  

# package imports
import numpy as np 
from scipy.integrate import  cumtrapz

# ----------------------------------------------------------------------
#  Lithium_Ion
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion(Battery):
    """ Generic lithium ion battery that specifies discharge/specific energy 
    characteristics. 
    
    Assumptions
    1) Default discharge curves correspond to lithium-iron-phosphate cells
    
    2) Convective Thermal Conductivity Coefficient corresponds to forced
    air cooling in 35 m/s air 
    
    Inputs:
    None
    
    Outputs:
    None
    
    Properties Used:
    N/A
    """  
    def __defaults__(self):
            
        self.tag               = 'Generic_Lithium_Ion_Battery_Cell'  
        self.cell              = Data()   
        self.module            = Data()        
        self.pack_config       = Data()
        self.module_config     = Data() 
        
        self.age                                           = 0       # [days]
        self.cell.mass                                     = None
        self.cell.charging_SOC_cutoff                      = 1. 
        self.cell.charging_current                         = 3.0     # [Amps]
        self.cell.charging_voltage                         = 3       # [Volts]
                         
        self.convective_heat_transfer_coefficient          = 35.     # [W/m^2K] 
        self.heat_transfer_efficiency                      = 1.0       
        
        self.pack_config.series                            = 1
        self.pack_config.parallel                          = 1  
        self.pack_config.total                             = 1   
        self.module_config.total                           = 1       
        self.module_config.normal_count                    = 1       # number of cells normal to flow
        self.module_config.parallel_count                  = 1       # number of cells parallel to flow      
        self.module_config.normal_spacing                  = 0.02
        self.module_config.parallel_spacing                = 0.02 
        
        self.cooling_fluid                                 = Air()    
        self.cooling_fluid.cooling_flowspeed               = 0.01       
        
        # defaults that are overwritten if specific cell chemistry is used 
        self.specific_energy                               = 200.    *Units.Wh/Units.kg    
        self.specific_power                                = 1.      *Units.kW/Units.kg
        self.ragone.const_1                                = 88.818  *Units.kW/Units.kg
        self.ragone.const_2                                = -.01533 /(Units.Wh/Units.kg)
        self.ragone.lower_bound                            = 60.     *Units.Wh/Units.kg
        self.ragone.upper_bound                            = 225.    *Units.Wh/Units.kg    
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
                  cell_charge_throughput                                   [Amp-hrs]
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
        Q_prior           = battery.cell_charge_throughput 
        R_growth_factor   = battery.R_growth_factor
        E_growth_factor   = battery.E_growth_factor 
        I                 = numerics.time.integrate
        D                 = numerics.time.differentiate

        if not battery_discharge_flag:   
            I_bat = -I_bat  
        # ---------------------------------------------------------------------------------
        # Compute battery electrical properties 
        # --------------------------------------------------------------------------------- 
         
        # Update battery capacitance (energy) with aging factor
        E_max = E_max*E_growth_factor
        
        # Compute state of charge and depth of discarge of the battery
        initial_discharge_state = np.dot(I,P_bat) + E_current[0]
        SOC_old                 = np.divide(initial_discharge_state,E_max)
        
        SOC_old[SOC_old>1] = 1.
        SOC_old[SOC_old<0] = 0.
        
        # Compute internal resistance
        R_bat = -0.0169*(SOC_old**4) + 0.0418*(SOC_old**3) - 0.0273*(SOC_old**2) + 0.0069*(SOC_old) + 0.0043
        R_0   = R_bat*R_growth_factor 
        R_0[R_0<0] = 0.  # when battery isn't being called
        
        # Compute Heat power generated by all cells
        Q_heat_gen = (I_bat**2.)*R_0
        
        # Determine actual power going into the battery accounting for resistance losses
        P = P_bat - np.abs(Q_heat_gen) 
        
        # Compute temperature rise of battery
        dT_dt     = Q_heat_gen /(bat_mass*bat_Cp)
        T_current = T_current[0] + np.dot(I,dT_dt)
        
        # Possible Energy going into the battery:
        energy_unmodified = np.dot(I,P)
    
        # Available capacity
        capacity_available = E_max - battery.current_energy[0]
    
        # How much energy the battery could be overcharged by
        delta           = energy_unmodified -capacity_available
        delta[delta<0.] = 0.
    
        # Power that shouldn't go in
        ddelta = np.dot(D,delta) 
    
        # Power actually going into the battery
        P[P>0.] = P[P>0.] - ddelta[P>0.]
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
        battery.cell_charge_throughput             = Q_total  
        battery.voltage_open_circuit               = V_oc
        battery.voltage_under_load                 = V_ul
        battery.current                            = I_bat 
        battery.pack_temperature                   = T_current 
        battery.cell_joule_heat_fraction           = np.zeros_like(V_ul)
        battery.cell_entropy_heat_fraction         = np.zeros_like(V_ul)  
        battery.cell_voltage_open_circuit          = np.zeros_like(V_ul)
        battery.cell_current                       = np.zeros_like(V_ul)
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
        segment.state.residuals.network.voltage = (v_predict - v_actual)/v_max
        
        return 
    
    def append_battery_unknowns_and_residuals_to_segment(self,segment,initial_voltage, 
                                              initial_battery_cell_temperature , initial_battery_state_of_charge,
                                              initial_battery_cell_current): 
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
                state   - segment unknowns to define voltage [unitless]
            
            Outputs
                V_ul    - under-load voltage                 [volts]
             
            Properties Used:
            N/A
        """              

        return state.conditions.propulsion.battery_voltage_under_load 
    
    def update_battery_state_of_health(self,segment,increment_battery_cycle_day = False):   
        print(' No aging model currently implemented for LFP cells. Pristine condition of \n '
              'the battery cell will be assigned each charge cycle')
        return  
 
  
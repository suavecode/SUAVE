## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion_LiNCA_18650.py
# 
# Created:  Feb 2020, M. Clarke
# Modified: Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
from SUAVE.Core import Units , Data 
from scipy.interpolate import  RectBivariateSpline
import numpy as np 
from SUAVE.Methods.Power.Battery.Discharge_Models.LiNCA_discharge   import LiNCA_discharge 
from SUAVE.Methods.Power.Battery.Charge_Models.LiNCA_charge         import LiNCA_charge 

## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
def Lithium_Ion_LiNCA_18650(self): 
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
    self.tag                              = 'Lithium_Ion_LiNCA_Cell' 
                                         
    self.cell.diameter                    = 0.01833  # [m]
    self.cell.height                      = 0.06485  # [m] 
    self.cell.surface_area                = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)  # [m^2]
    self.cell.volume                      = np.pi*(0.5*self.cell.mass)**2*self.cell.height
    self.cell.density                     = 760       # [kg/m^3]  
    self.cell.mass                        = 0.048 * Units.kg 
    self.cell.electrode_area              = 0.0342    # [m^2] 
                                          
    self.cell.max_voltage                 = 4.2   # [V]
    self.cell.nominal_capacity            = 3.00  # [Amp-Hrs]
    self.cell.nominal_voltage             = 3.6   # [V]
    self.cell.charging_voltage            = self.cell.nominal_voltage   # [V]  
    
    self.watt_hour_rating                 = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
    self.specific_energy                  = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
    self.specific_power                   = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
    self.resistance                       = 0.025   # [Ohms]
                                          
    self.specific_heat_capacity           = 837.4   # [J/kgK] 
    self.cell.specific_heat_capacity      = 837.4   # [J/kgK]   
    self.cell.radial_thermal_conductivity = 0.8     # [J/kgK]  
    self.cell.axial_thermal_conductivity  = 32.2    # [J/kgK]  
       
    self.discharge_model                  = LiNCA_discharge 
    self.charge_model                     = LiNCA_charge 
    battery_raw_data                      = load_NCA_raw_results()                                                   
    self.discharge_performance_map        = create_discharge_performance_map(battery_raw_data)        
    return  

def create_discharge_performance_map(battery_raw_data):
    """ Create discharge and charge response surface for 
        LiNCA  battery cells using raw data     
        
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
    return SUAVE.Input_Output.SUAVE.load('NCA_Raw_Data.res') 
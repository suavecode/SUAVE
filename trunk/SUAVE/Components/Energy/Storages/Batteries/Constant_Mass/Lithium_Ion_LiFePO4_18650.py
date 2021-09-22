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
from SUAVE.Methods.Power.Battery.Discharge_Models.LiFePO4_discharge   import LiFePO4_discharge
from SUAVE.Methods.Power.Battery.Charge_Models.LiFePO4_charge         import LiFePO4_charge
import numpy as np 
 
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
def Lithium_Ion_LiFePO4_18650(self):
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
    self.tag                              = 'Lithium_Ion_LiFePO4_Cell' 
     
    self.cell.mass                        = 30 * Units.kg  
    self.cell.diameter                    = 0.018   # [m]
    self.cell.height                      = 0.0650   # [m] 
    self.cell.surface_area                = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)  # [m^2]
    self.cell.volume                      = np.pi*(0.5*self.cell.diameter)**2*self.cell.height  # [m^3] 
    self.cell.density                     = self.cell.mass/self.cell.volume  # [kg/m^3]
    self.cell.electrode_area              = 0.0342  # [m^2]  # estimated 
   
    self.cell.max_voltage                 = 3.2  # [V]
    self.cell.nominal_capacity            = 1.3  # [Amp-Hrs]
    self.cell.nominal_voltage             = 3.2  # [V]
    self.cell.charging_voltage            = self.cell.nominal_voltage   # [V]  
     
    self.watt_hour_rating                 = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
    self.specific_energy                  = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
    self.specific_power                   = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
    self.ragone.const_1                   = 88.818  * Units.kW/Units.kg
    self.ragone.const_2                   = -.01533 / (Units.Wh/Units.kg)
    self.ragone.lower_bound               = 60.     * Units.Wh/Units.kg
    self.ragone.upper_bound               = 225.    * Units.Wh/Units.kg         
    self.resistance                       = 0.022   # [Ohms]
  
    self.specific_heat_capacity           = 1115    # [J/kgK] 
    self.cell.specific_heat_capacity      = 1115    # [J/kgK] 
    self.cell.radial_thermal_conductivity = 0.475   # [J/kgK]  
    self.cell.axial_thermal_conductivity  = 37.6    # [J/kgK]  
       
     
    self.discharge_model                   = LiFePO4_discharge
    self.charge_model                      = LiFePO4_charge  
    
    return  
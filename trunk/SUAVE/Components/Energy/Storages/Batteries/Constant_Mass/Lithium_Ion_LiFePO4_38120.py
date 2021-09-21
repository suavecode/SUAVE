## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Ion_LiFePO4_38120.py
# 
# Created:  Feb 2020, M. Clarke
# Modified: Sep 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
from SUAVE.Core import Units 
from SUAVE.Components.Energy.Storages.Batteries.Constant_Mass         import Lithium_Ion 
from SUAVE.Methods.Power.Battery.Discharge_Models.LiFePO4_discharge   import LiFePO4_discharge
from SUAVE.Methods.Power.Battery.Charge_Models.LiFePO4_charge         import LiFePO4_charge
import numpy as np 
 
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion_LiFePO4_38120(Lithium_Ion):
    """ Specifies discharge/specific energy characteristics specific 
        18650 lithium-iron-phosphate-oxide battery cells     
        
        Assumptions:
        Convective Thermal Conductivity Coefficient corresponds to forced
        air cooling in 35 m/s air 
        
        Source:
        Saw, Lip Huat, et al. "Computational fluid dynamic and thermal analysis of 
        Lithium-ion battery pack with air cooling." Applied energy 177 (2016): 783-792.
        
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
        self.tag                                          = 'Lithium_Ion_LiFePO4_Cell' 
        
        self.cell.mass                                    = 0.335  * Units.kg 
        self.cell.diameter                                = 0.038   # [m]
        self.cell.height                                  = 0.146   # [m] 
        self.cell.surface_area                            = (np.pi*self.cell.height*self.cell.diameter) + (0.5*np.pi*self.cell.diameter**2)  # [m^2]
        self.cell.volume                                  = self.cell.height*0.25*np.pi*(self.cell.diameter**2)     # [m^3] 
        self.cell.density                                 = self.cell.diameter/self.cell.volume  # [kg/m^3] 
        self.cell.electrode_area                          = 0.07    # [m^2] 
     
        self.cell.max_voltage                             = 3.8 # [V]
        self.cell.nominal_capacity                        = 8.0 # [Amp-Hrs]
        self.cell.nominal_voltage                         = 3.3 # [V]
        self.cell.charging_voltage                        = self.cell.nominal_voltage   # [V]  
        
        self.watt_hour_rating                             = self.cell.nominal_capacity  * self.cell.nominal_voltage  # [Watt-hours]      
        self.specific_energy                              = self.watt_hour_rating*Units.Wh/self.cell.mass            # [J/kg]
        self.specific_power                               = self.specific_energy/self.cell.nominal_capacity          # [W/kg]   
        self.ragone.const_1                               = 88.818  * Units.kW/Units.kg
        self.ragone.const_2                               = -.01533 / (Units.Wh/Units.kg)
        self.ragone.lower_bound                           = 60.     * Units.Wh/Units.kg
        self.ragone.upper_bound                           = 225.    * Units.Wh/Units.kg         
        self.resistance                                   = 0.0034   # [Ohms]
     
        self.specific_heat_capacity                       = 998     # [J/kgK] 
        self.cell.specific_heat_capacity                  = 998     # [J/kgK] 
        self.cell.thermal_conductivity                    = 32.2    # [J/kgK]   
          
        
        self.discharge_model                               = LiFePO4_discharge
        self.charge_model                                  = LiFePO4_charge  
        
        return  
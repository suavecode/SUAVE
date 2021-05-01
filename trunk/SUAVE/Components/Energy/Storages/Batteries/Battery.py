## @ingroup Components-Energy-Storages-Batteries
# Battery.py
# 
# Created:  Nov 2014, M. Vegh
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Battery
# ----------------------------------------------------------------------    

## @ingroup Components-Energy-Storages-Batteries
class Battery(Energy_Component):
    """
    Energy Component object that stores energy. Contains values
    used to indicate its discharge characterics, including a model
    that calculates discharge losses
    """
    def __defaults__(self):
        self.mass_properties.mass = 0.0
        self.energy_density       = 0.0
        self.current_energy       = 0.0
        self.resistance           = 0.07446 #base internal resistance of battery in ohms
        self.max_energy           = 0.0
        self.max_power            = 0.0
        self.max_voltage          = 0.0
        self.discharge_model      = SUAVE.Methods.Power.Battery.Discharge.datta_discharge
        self.ragone               = Data()
        self.ragone.const_1       = 0.0 #used for ragone functions; 
        self.ragone.const_2       = 0.0 #specific_power=ragone_const_1*10^(specific_energy*ragone_const_2)
        self.ragone.lower_bound   = 0.0 #lower bound specific energy for which ragone curves no longer make sense
        self.ragone.upper_bound   = 0.0
        
    def energy_calc(self,numerics):
        self.discharge_model(self, numerics)
        return  
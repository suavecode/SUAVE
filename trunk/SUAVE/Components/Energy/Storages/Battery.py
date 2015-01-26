#Battery.py
# 
# Created:  M Vegh, November 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
from SUAVE.Attributes import Units
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Power.Battery.Discharge.datta_discharge import datta_discharge
from SUAVE.Structure import Data

# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Battery(Energy_Component):
    
    def __defaults__(self):
        self.mass_properties.mass = 0.0
        self.energy_density       = 0.0
        self.current_energy       = 0.0
        self.resistance           = 0.0
        self.max_energy           = 0.0
        self.max_power            = 0.0
        self.discharge_model      = datta_discharge
        self.ragone               = Data()
        self.ragone.const_1       = 0.0 #used for ragone functions; 
        self.ragone.const_2       = 0.0 #specific_power=ragone_const_1*10^(specific_energy*ragone_const_2)
        self.ragone.lower_bound   = 0.0 #lower bound specific energy for which ragone curves no longer make sense
        self.ragone.upper_bound   = 0.0
        
    def energy_calc(self,numerics):
        self.discharge_model(self.numerics)
        return  
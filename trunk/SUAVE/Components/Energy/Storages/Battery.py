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
from SUAVE.Methods.Power.Battery.Discharge import datta_discharge
# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Battery(Energy_Component):
    
    def __defaults__(self):
        
        self.type = 'Li-Ion'
        self.mass_properties.mass = 0.0
        self.energy_density       = 0.0
        self.current_energy       = 0.0
        self.resistance           = 0.0
        self.max_energy           = 0.0
        self.max_power            = 0.0
        self.discharge_model      = datta_discharge
        
     
    def energy_calc(self,numerics):
        self.discharge_model(self.numerics)
        return  
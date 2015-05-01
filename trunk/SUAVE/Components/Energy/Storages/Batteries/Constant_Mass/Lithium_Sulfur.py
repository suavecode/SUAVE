#Battery.py
# 
# Created:  Michael Vegh, November 2014
# Modified:  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
import numpy as np
import scipy as sp
from SUAVE.Core                        import Units
from SUAVE.Components.Energy.Storages.Batteries  import Battery
# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Lithium_Sulfur(Battery):
    
    def __defaults__(self):
        self.specific_energy=500*Units.Wh/Units.kg
        self.specific_power=1*Units.kW/Units.kg
        self.ragone.const_1=245.848*Units.kW/Units.kg
        self.ragone.const_2=-.00478/(Units.Wh/Units.kg)
        self.ragone.lower_bound=300*Units.Wh/Units.kg
        self.ragone.upper_bound=700*Units.Wh/Units.kg
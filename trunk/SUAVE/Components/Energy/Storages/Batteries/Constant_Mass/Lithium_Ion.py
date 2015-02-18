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
from SUAVE.Core                         import Units
from SUAVE.Components.Energy.Storages.Batteries  import Battery
# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Lithium_Ion(Battery):
    
    def __defaults__(self):
        self.specific_energy=90*Units.Wh/Units.kg
        self.specific_power=1*Units.kW/Units.kg
        self.ragone.const_1=88.818*Units.kW/Units.kg
        self.ragone.const_2=-.01533/(Units.Wh/Units.kg)
        self.ragone.lower_bound=60*Units.Wh/Units.kg
        self.ragone.upper_bound=225*Units.Wh/Units.kg
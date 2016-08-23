# Lithium_Ion.py
# 
# Created:  Nov 2014, M. Vegh
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Core import Units
from SUAVE.Components.Energy.Storages.Batteries  import Battery

# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Lithium_Ion(Battery):
    
    def __defaults__(self):
        self.specific_energy    = 200.    *Units.Wh/Units.kg
        self.specific_power     = 1.      *Units.kW/Units.kg
        self.ragone.const_1     = 88.818  *Units.kW/Units.kg
        self.ragone.const_2     = -.01533 /(Units.Wh/Units.kg)
        self.ragone.lower_bound = 60.     *Units.Wh/Units.kg
        self.ragone.upper_bound = 225.    *Units.Wh/Units.kg
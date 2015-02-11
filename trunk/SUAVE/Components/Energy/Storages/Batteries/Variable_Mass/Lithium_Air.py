#Battery.py
# 
# Created:  Michael Vegh
# Modified: October, 2014

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports

from SUAVE.Attributes import Units
from SUAVE.Components.Energy.Storages.Batteries  import Battery

# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Lithium_Air(Battery):
    
    def __defaults__(self):
        self.specific_energy=2000.*Units.Wh/Units.kg    #convert to Joules/kg
        self.specific_power=0.66*Units.kW/Units.kg      #convert to W/kg
        self.mass_gain_factor=(1.92E-4)/Units.Wh
       
            
  
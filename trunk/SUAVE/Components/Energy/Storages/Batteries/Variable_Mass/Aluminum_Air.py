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
from SUAVE.Core import Units
from SUAVE.Components.Energy.Storages.Batteries  import Battery
# ----------------------------------------------------------------------
#  Battery Class
# ----------------------------------------------------------------------    

class Aluminum_Air(Battery):
    
    def __defaults__(self):
        self.specific_energy=1300.*Units.Wh/Units.kg    #convert to Joules/kg
        self.specific_power=0.2*Units.kW/Units.kg      #convert to W/kg
        self.mass_gain_factor=(0.000220289)/Units.Wh
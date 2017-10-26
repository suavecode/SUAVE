## @ingroup Components-Energy-Storages-Batteries-Variable_Mass
# Lithium_Air.py
# 
# Created:  Oct 2014, M. Vegh
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
## @ingroup Components-Energy-Storages-Batteries-Variable_Mass
class Lithium_Air(Battery):
    """
    Specifies specific energy characteristics specific to
    lithium-air batteries. Also includes parameters related to 
    consumption of oxygen
    """
    
    
    def __defaults__(self):
        self.specific_energy  = 2000.     *Units.Wh/Units.kg    # convert to Joules/kg
        self.specific_power   = 0.66      *Units.kW/Units.kg    # convert to W/kg
        self.mass_gain_factor = (1.92E-4) /Units.Wh
       
            
  
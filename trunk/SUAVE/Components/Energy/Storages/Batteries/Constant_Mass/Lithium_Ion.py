## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
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
from SUAVE.Core import Units , Data 
from SUAVE.Components.Energy.Storages.Batteries  import Battery

# ----------------------------------------------------------------------
#  Lithium_Ion
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Ion(Battery):
    """
    Specifies discharge/specific energy characteristics of the default 
    lithium-ion battery in SUAVE. 
    """
    def __defaults__(self):
        self.cell                        = Data()
        self.pack_config               = Data()
        
        self.specific_energy             = 200.    * Units.Wh/Units.kg
        self.specific_power              = 1.      * Units.kW/Units.kg
        self.ragone.const_1              = 88.818  * Units.kW/Units.kg
        self.ragone.const_2              = -.01533 / (Units.Wh/Units.kg)
        self.ragone.lower_bound          = 60.     * Units.Wh/Units.kg
        self.ragone.upper_bound          = 225.    * Units.Wh/Units.kg
        
        self.pack_config.series        = 1
        self.pack_config.parallel      = 1   
        self.specific_heat_capacity      = 2000 
        self.heat_transfer_coefficient   = 35.  
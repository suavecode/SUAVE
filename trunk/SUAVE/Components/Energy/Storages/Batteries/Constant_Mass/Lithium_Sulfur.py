## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
# Lithium_Sulfur.py
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
#  Lithium_Sulfur
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Storages-Batteries-Constant_Mass
class Lithium_Sulfur(Battery):
    """
    Specifies discharge/specific energy characteristics specific to
    lithium-ion batteries
    """
    
    def __defaults__(self):
        self.specific_energy    = 500     *Units.Wh/Units.kg
        self.specific_power     = 1       *Units.kW/Units.kg
        self.ragone.const_1     = 245.848 *Units.kW/Units.kg
        self.ragone.const_2     = -.00478 /(Units.Wh/Units.kg)
        self.ragone.lower_bound = 300     *Units.Wh/Units.kg
        self.ragone.upper_bound = 700     *Units.Wh/Units.kg
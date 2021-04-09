## @ingroup Components-Energy-Storages-Fuel_Tanks
# Fuel_Tank.py
# 
# Created:  Sep 2018, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Fuel Tank
# ----------------------------------------------------------------------    

## @ingroup Components-Energy-Storages-Fuel_Tank
class Fuel_Tank(Energy_Component):
    """
    Energy Component object that stores fuel. Contains values
    used to indicate its fuel type.
    """
    def __defaults__(self):
        self.mass_properties.empty_mass            = 0.0
        self.mass_properties.fuel_mass_when_full   = 0.0
        self.mass_properties.fuel_volume_when_full = 0.0
        self.fuel_type                             = None
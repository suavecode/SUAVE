## @ingroup Components-Energy-Storages-Cryogen_Tanks
# Cryogen_Tank.py
# 
# Created:  Jan 2020, K.Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports

from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Cryogen_Tank
# ----------------------------------------------------------------------    

## @ingroup Components-Energy-Storages-Cryogen_Tank
class Cryogen_Tank(Energy_Component):
    """
    Energy Component object that stores cryogen. Contains values
    used to indicate its cryogen type.
    """
    def __defaults__(self):
        self.mass_properties.empty_mass                 = 0.0
        self.mass_properties.cryogen_mass_when_full     = 0.0
        self.mass_properties.cryogen_volume_when_full   = 0.0
        self.cryogen_type                               = None
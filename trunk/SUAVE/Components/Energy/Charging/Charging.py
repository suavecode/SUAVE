## @ingroup Components-Energy-Charging
# Charging.py
# 
# Created: Aug. 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Charging Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Charging
class Charging(Energy_Component):
    """A class representing Charging.
    
    Assumptions:
    None
    
    Source:
    N/A
    """        
    def __defaults__(self):
        """This sets the default charging C rate and SOC cutoff when battery is fully charged.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """                  
        self.C_rate     = 5. 
        self.SOC_cutoff = 1.0       
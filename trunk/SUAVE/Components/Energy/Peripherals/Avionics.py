## @ingroup Components-Energy-Peripherals
# Avionics.py
# 
# Created:  Jun 2014, E. Botero
# Modified: Feb 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  Avionics Class
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Peripherals
class Avionics(Energy_Component):
    """A class representing avionics.
    
    Assumptions:
    None
    
    Source:
    N/A
    """        
    def __defaults__(self):
        """This sets the default power draw.

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
        self.power_draw = 0.0
        self.tag        = 'avionics'
        
    def power(self):
        """This gives the power draw from avionics.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        self.outputs.power_draw

        Properties Used:
        self.power_draw
        """                 
        self.outputs.power = self.power_draw
        
        return self.power_draw
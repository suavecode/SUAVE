## @ingroup Components-Energy-Converters
# Solar_Panel.py
#
# Created:  Jun 2014, E. Botero
# Modified: Jan 2016, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Components.Energy.Energy_Component import Energy_Component
## @ingroup Components-Energy-Converters
# ----------------------------------------------------------------------
#  Solar_Panel Class
# ----------------------------------------------------------------------
class Solar_Panel(Energy_Component):
    """This is a solar cell component.
    
    Assumptions:
    None

    Source:
    None
    """       
    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        None

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """        
        self.area       = 0.0
        self.efficiency = 0.0
    
    def power(self):
        """This determines the power output of the solar cell.

        Assumptions:
        None

        Source:
        None

        Inputs:
        self.inputs.flux   [W/m^2]

        Outputs:
        self.outputs.power [W]
        power              [W]

        Properties Used:
        self.efficiency    [-]
        self.area          [m^2]
        """        
        # Unpack
        flux       = self.inputs.flux
        efficiency = self.efficiency
        area       = self.area
        
        p = flux*area*efficiency
        
        # Store to outputs
        self.outputs.power = p
    
        return p
    
    
    
    
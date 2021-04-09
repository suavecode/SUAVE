## @ingroup Components-Energy-Converters
# Generator_Zero_Fid.py
#
# Created:  Apr 2019, C. McMillan


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Components.Energy.Energy_Component import Energy_Component
import numpy as np

# ----------------------------------------------------------------------
#  Generator Class
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Generator_Zero_Fid(Energy_Component):
    """This is a zero-fidelity generator component.
    
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
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """              
        self.sfc        = 0.0
        self.max_power  = 0.0

    
    def calculate_power(self,conditions):
        """Calculates power 
    
        Assumptions:
        Specific Fuel Consumption (SFC) is constant
        Generator always operates at max power which is constant
        conditions.propulsion.throttle exists
    
        Source:
        N/A
    
        Inputs:
        N/A
    
        Outputs:
        self.outputs.
            power_generated    [W]
            mdot               [kg/s]
    
        Properties Used:
        self.
            sfc                [kg/(W*s)]
            max_power          [W]

        """       
        
        # Unpack
        sfc         = self.sfc
        max_power   = self.max_power

        mdot = max_power * sfc
        
        # Pack
        self.outputs.power_generated = max_power * np.ones_like(conditions.propulsion.throttle)
        self.outputs.vehicle_mass_rate = mdot * np.ones_like(conditions.propulsion.throttle)
        
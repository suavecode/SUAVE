## @ingroup Components-Energy-Distributors
# HTS_DC_Supply.py
#
# Created:  Feb 2020,   K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004 
# Modified: Nov 2021,   S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
import MARC

from MARC.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  HTS DC Supply Class
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Distributors
class HTS_DC_Supply(Energy_Component):
    
    def __defaults__(self):
        """ This sets the default values.
    
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
        
        self.efficiency         =   0.0
        self.rated_current      = 100.0     # [A]
        self.rated_power        = 100.0     # [W]
        self.inputs.power_out   = 0.0       # [W]
        self.outputs.power_in   = 0.0       # [W]
    
    def power(self, conditions):
        """ The power that must be supplied to the DC supply to power the HTS coils.

            Assumptions:
            Supply cable is solid copper, i.e. not a rotating joint.
            Power supply has static efficiency across current output range.
            Power supply performance is not affected by altitude or other environmental factors. This is not generally true (Ametek SGe datasheet should be derated by 10% per 1000 feet) for current supplies designed for ground use however a supply specifically designed for airborne use can be expected to have a more appropriate cooling design that would allow high altitude use.

            Source:
            N/A

            Inputs:
            self.inputs
                current             [A]
                power_out           [W]

            Outputs:
            self.outputs
                power_in            [W]

            Properties Used:
            self. 
                efficiency 

        """
        # Unpack
        power_out               = self.inputs.power_out
        efficiency              = self.efficiency

        # Apply the efficiency of the current supply to get the total power required at the input of the current supply.
        power_in                = power_out/efficiency


        # Return basic result.
        self.outputs.power_in = power_in

        return power_in
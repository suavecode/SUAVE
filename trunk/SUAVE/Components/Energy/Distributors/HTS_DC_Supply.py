## @ingroup Components-Energy-Distributors
# HTS_DC_Supply.py
#
# Created:  Feb 2020,   K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

from SUAVE.Components.Energy.Energy_Component import Energy_Component

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
        self.mass               = 100.0     # [kg]
        self.rated_current      = 100.0     # [A]
        self.rated_power        = 100.0     # [W]
    
    def power(self, current, power_out):
        """ The power that must be supplied to the DC supply to power the HTS coils.

            Assumptions:
            Supply cable is solid copper, i.e. not a rotating joint.
            Power supply has static efficiency across current output range.

            Source:
            N/A

            Inputs:
            current             [A]
            power_out           [W]
            self.efficiency

            Outputs:
            power_in            [W]

        """
        # Unpack
        efficiency              = self.efficiency

        # Apply the efficiency of the current supply to get the total power required at the input of the current supply.
        power_in                = power_out/efficiency

        # # Store output values.
        # self.output.power_in    = power_in

        # Return basic result.
        return power_in
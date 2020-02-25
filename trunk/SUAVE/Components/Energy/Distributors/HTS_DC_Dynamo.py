## @ingroup Components-Energy-Distributors
# HTS_DC_Dynamo.py
#
# Created:  Feb 2020,   K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  HTS DC Dynamo Class
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Distributors
class HTS_DC_Dynamo(Energy_Component):
    
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
        self.rated_RPM          = 100.0     # [RPM]
        self.rated_temp         =  77.0     # [K]
    
    def power(self, cryo_temp, current, power_out):
        """ The shaft power that must be supplied to the DC Dynamo supply to power the HTS coils.

            Assumptions:
            Dynamo ESC has static efficiency across speed and power output ranges.

            Source:
            N/A

            Inputs:
            cryo_temp           [K]
            current             [A]
            power_out           [W]
            self.
                efficiency
                rated_current   [A]
                rated_RPM       [RPM]
                rated_temp      [K]

            Outputs:
            power_in            [W]
            cryo_load           [W]

        """
        # Unpack
        efficiency              = self.efficiency
        rated_current           = self.rated_current
        rated_temp              = self.rated_temp   

        if current != rated_current:
            print("Warning, HTS Dynamo not operating at rated current, input power underestimated.")

        if cryo_temp != rated_temp:
            print("Warning, HTS dynamo not operating at rated temperature. Ensure operating temperature is below T_c.")

        # When inactive the dynamo does not put a load on the cryogenic system
        if power_out == 0.0:
            power_in    = 0.0
            cryo_load   = 0.0
            return [power_in, cryo_load]

        # Otherwise, for now, assume dynamo is operating at the rated efficiency.
        power_in = power_out/efficiency

        # Calculate the dynamo losses. This loss will directly heat the cryogenic environment.
        cryo_load = power_in - power_out

        # Return basic results.
        return [power_in, cryo_load]


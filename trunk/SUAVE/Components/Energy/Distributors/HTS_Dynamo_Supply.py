## @ingroup Components-Energy-Distributors
# HTS_Dynamo_Supply.py
#
# Created:  Feb 2020,   K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component

# ----------------------------------------------------------------------
#  HTS Dynamo Supply Class
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Distributors
class HTS_Dynamo_Supply(Energy_Component):
    """ This supplies the HTS dynamo with shaft power.
        Technically this is a converter, however this is stored in Distributors as it in analagous to the DC Power Supply for the HTS coils, and when used in combination with the HTS dynamo the power supplied to this component electrically is delivered as electric power to the HTS coil.
        Practically this component is a ESC, a DC electric motor, and an attached gearbox, for example a Maxxon GP32 Planetary Gearbox, Maxxon EC-max 30 Brushless Motor, and a Maxxon DEC 50/5 Speed Controller.
    """

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
        
        self.efficiency     =    0.0
        self.mass           =  100.0
        self.rated_RPM      = 1000.0
    
    def power_in(self, power_out, RPM=None):
        """ The power supplied to this component based on that that this must deliver to the HTS dynamo as shaft power.

            Assumptions:
            Constant efficiency across the RPM range of the output shaft.

            Source:
            N/A

            Inputs:
            RPM                 [RPM]
            power_out           [W]
            self.efficiency

            Outputs:
            power_in            [W]

        """
        # Unpack
        efficiency  = self.efficiency
        rated_RPM   = self.rated_RPM

        # Assume rated RPM is no RPM value supplied
        if RPM == None:
            RPM = rated_RPM

        # Create output array
        power_in    = np.zeros_like(power_out)

        # Apply the efficiency of the current supply to get the total power required at the input of the current supply. For more precise results efficiency could be adjusted based on RPM.
        power_in                = power_out/efficiency

        # Store output values.
        self.output.power_in    = power_in

        # Return basic result.
        return power_in


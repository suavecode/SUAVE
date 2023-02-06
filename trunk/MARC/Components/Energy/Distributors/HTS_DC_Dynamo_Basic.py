## @ingroup Components-Energy-Distributors
# HTS_DC_Dynamo_Basic.py
#
# Created:  Feb 2020,   K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004 
# Modified: Jan 2022,   S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
import MARC
import numpy as np
from MARC.Components.Energy.Energy_Component import Energy_Component
from MARC.Methods.Cryogenics.Dynamo.dynamo_efficiency import efficiency_curve
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  HTS DC Dynamo Class
# ----------------------------------------------------------------------

## @ingroup Components-Energy-Distributors
class HTS_DC_Dynamo_Basic(Energy_Component):
    """ Basic HTS Dynamo model for constant current DC coil operation at constant cryogenic temperature.

        Assumptions:
        HTS Dynamo is operating at rated temperature and output current.

        Source:
        None
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
        
        self.efficiency                     =   0.0      # [W/W]
        self.mass_properties.mass           =   0.0      # [kg] 
        self.rated_current                  =   0.0      # [A]
        self.rated_RPM                      =   0.0      # [RPM]
        self.rated_temp                     =   0.0      # [K]
        self.inputs.hts_current             =   0.0      # [A]
        self.inputs.power_out               =   0.0      # [W]
        self.outputs.cryo_load              =   0.0      # [W]
        self.outputs.power_in               =   0.0      # [W]

    
    def shaft_power(self, conditions):
        """ The shaft power that must be supplied to the DC Dynamo supply to power the HTS coils.
            Assumptions:
                HTS Dynamo is operating at rated temperature.
                
            Source:
                N/A

            Inputs:
            self.inputs
                hts_current     [A]
                power_out       [W]

            Outputs:
            self.outputs.
                power_in            [W]
                cryo_load           [W]

            Properties Used:
            self. 
                rated_current       [A]
                efficiency          

        """

        hts_current = self.inputs.hts_current 

        power_out   = self.inputs.power_out 

        #Adjust efficiency according to the rotor current 
        current    = np.array(hts_current)
        efficiency = efficiency_curve(self, current)

        # Create output arrays. The hts dynamo input power is assumed zero if the output power is zero, this may not be true for some dynamo configurations however the power required for zero output power will be very low.
        # Similarly, the cryo load will be zero if no dynamo effect is occuring.
        power_in = np.array(power_out/efficiency)
        power_in[power_out==0.] = 0
        
        cryo_load  = np.array(power_in - power_out)

        # Return basic results.

        self.outputs.cryo_load              =   cryo_load
        self.outputs.power_in               =   power_in

        return [power_in, cryo_load]





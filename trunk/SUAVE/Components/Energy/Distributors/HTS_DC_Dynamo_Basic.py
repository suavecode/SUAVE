## @ingroup Components-Energy-Distributors
# HTS_DC_Dynamo_Basic.py
#
# Created:  Feb 2020,   K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component
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
        
        self.efficiency             =   0.16
        self.mass_properties.mass   = 100.0     # [kg]
        self.rated_current          = 800.0     # [A]
        self.rated_RPM              = 120.0     # [RPM]
        self.rated_temp             =  77.0     # [K]
    
    def shaft_power(self, cryo_temp, hts_current, power_out):
        """ The shaft power that must be supplied to the DC Dynamo supply to power the HTS coils.
            Assumptions:
            HTS Dynamo is operating at rated current and temperature.
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
        rated_current           = self.rated_current
        rated_temp              = self.rated_temp

        #Adjust efficiency according to the rotor current 
        efficiency = self.efficiency_curve(hts_current)

        # Create output arrays. The hts dynamo input power is assumed zero if the output power is zero, this may not be true for some dynamo configurations however the power required for zero output power will be very low.
        # Similarly, the cryo load will be zero if no dynamo effect is occuring.
        power_in            = np.zeros_like(power_out)
        cryo_load           = np.zeros_like(power_out)

        # force hts current to be an array if it isn't already

        print("hts current = ", hts_current)
        if type(hts_current) == float:
            current     = hts_current
            hts_current = np.ones_like(power_out) * current
            print("hts current now = ", hts_current)

        # force cryogenic temperature to be an array if it isn't already
        if type(cryo_temp) == float:
            temp     = cryo_temp
            cryo_temp = np.ones_like(power_out) * temp

        # Iterate through the operating condition arrays
        for index, power in np.ndenumerate(power_out):
            # In this basic model the current and operating temperature are assumed constant, so warn if this is not true
            #if hts_current[index] != rated_current:
                #print("Warning, HTS Dynamo not operating at rated current, input power underestimated.")
            #if cryo_temp[index] != rated_temp:
                #print("Warning, HTS dynamo not operating at rated temperature. Ensure operating temperature is below T_c.")

            # In this basic model, assume dynamo is operating at the rated efficiency.
            power_in[index] = power/efficiency[index]


            # Calculate the dynamo losses. This loss will directly heat the cryogenic environment.
            cryo_load[index] = power_in[index] - power

        # Return basic results.
        return [power_in, cryo_load]

    def efficiency_curve(self, x):

        """ This sets the default values.
        Assumptions:
        None
        Source:
        N/A
        LINK TO KENTS STUFF
        Inputs:
        None
        Outputs:
        None
        Properties Used:
        None
        """     

        x = np.array(x)

        if np.any(x > self.rated_current * 1.8 ) or np.any(x < self.rated_current * 0.2): #Plus minus 80
            print("out of bounds")

        a = ( self.efficiency ) / np.square(self.rated_current) #one point on the graph is assumed to be  (0, 2 * current), 0  = a (current ^ 2) + efficiency 
        
        return   -a * (np.square( x - self.rated_current) ) +  self.efficiency # y = a(x - current)^2 + efficieny 
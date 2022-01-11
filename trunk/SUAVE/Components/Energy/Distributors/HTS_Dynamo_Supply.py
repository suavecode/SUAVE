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
        
        self.efficiency             =    0.0
        self.mass_properties.mass   =    0.0    # [kg]
        self.rated_RPM              =    0.0    # [RPM]
    
    def power_in(self, dynamo, power_out, hts_current, RPM=None):
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
        rated_RPM   = self.rated_RPM
        
        #Adjust efficiency according to the rotor current 
        efficiency  = dynamo.efficiency_curve(hts_current)
        

        # Assume rated RPM is no RPM value supplied
        if RPM == None:
            RPM = rated_RPM

        # Create output array
        power_in    = np.zeros_like(power_out)

        # Apply the efficiency of the current supply to get the total power required at the input of the current supply. For more precise results efficiency could be adjusted based on RPM.
        power_in    = power_out/efficiency

        # Return basic result.
        return power_in



    def mass_estimation(self):
        """ Basic mass estimation for HTS Dynamo supply. This supply includes all elements required to create the required shaft power from supplied electricity, i.e. the esc, brushless motor, and gearbox.
        Assumptions:
        Mass scales linearly with power and current
        Source:
        Maxon Motor drivetrains
        Inputs:
        current             [A]
        power_out           [W]
        Outputs:
        mass                [kg]
        """

        # unpack
        rated_power     = self.rated_power

        # Estimate mass of motor and gearbox. Source: Maxon EC-max 12V brushless motors under 100W.
        mass_motor      = 0.013 + 0.0046 * rated_power
        mass_gearbox    = 0.0109 + 0.0015 * rated_power

        # Estimate mass of motor driver (ESC). Source: Estimate
        mass_esc        = (5.0 + rated_power/50.0)/1000.0

        # Sum masses to give total mass
        mass            = mass_esc + mass_motor + mass_gearbox

        # Store results
        self.mass_properties.mass       = mass

        # Return results
        return mass
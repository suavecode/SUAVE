## @ingroup Components-Energy-Distributors
# HTS_Dynamo_Supply.py
#
# Created:  Feb 2020,   K. Hamilton - Through New Zealand Ministry of Business Innovation and Employment Research Contract RTVU2004 
# Modified: Feb 2022,   S. Claridge 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE
import numpy as np
from SUAVE.Components.Energy.Energy_Component import Energy_Component

from SUAVE.Methods.Cryogenics.Dynamo.dynamo_efficiency import efficiency_curve

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
        
        self.efficiency                     =    0.0
        self.mass_properties.mass           =    0.0    # [kg]
        self.rated_RPM                      =    0.0    # [RPM]
        self.inputs.dynamo                  =    None
        self.inputs.power_out               =    0.0    #[W]
        self.inputs.hts_current             =    0.0    #[A]
        self.inputs.RPM                     =    0.0    # [RPM]
        self.outputs.power_in               =    0.0    #[W] 


    
    def power_in(self, conditions):
        """ The power supplied to this component based on that this must deliver to the HTS dynamo as shaft power.

            Assumptions:
                Constant efficiency across the RPM range of the output shaft.

            Source:
                N/A

            Inputs:
            self.inputs.
                RPM                 [RPM]
                power_out           [W]
                hts_current         [A]

            Outputs:
            self.outputs.
                power_in            [W]

            Properties Used:
            self.
                rated_RPM
        """
        # Unpack
        RPM         = self.inputs.RPM
        dynamo      = self.inputs.dynamo
        hts_current = self.inputs.hts_current
        power_out   = self.inputs.power_out
        
        #Adjust efficiency according to the rotor current 
        efficiency  = efficiency_curve(dynamo, hts_current)
        

        # Assume rated RPM is no RPM value supplied
        if RPM == 0.0:
            RPM = self.rated_RPM 

        # Create output array
        power_in    = np.zeros_like(power_out)

        # Apply the efficiency of the current supply to get the total power required at the input of the current supply. For more precise results efficiency could be adjusted based on RPM.
        power_in    = power_out/efficiency

        # Return basic result.
        self.outputs.power_in = power_in
        return power_in



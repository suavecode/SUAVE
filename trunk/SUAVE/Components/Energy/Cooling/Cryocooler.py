## @ingroup Components-Energy-Cooling
# Cryocooler.py
# 
# Created:  Feb 2020,   K.Hamilton
# Modified: Nov 2021,   S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Cryogenics.Cryocooler.cryocooler_model import cryocooler_model
import numpy as np

# ----------------------------------------------------------------------
#  Cryocooler
# ----------------------------------------------------------------------    
## @ingroup Components-Energy-Cooling-Cryocooler
class Cryocooler(Energy_Component):
    
    """
    Cryocooler provides cooling power to cryogenic components.
    Energy is used by this component to provide the cooling, despite the cooling power provided also being an energy inflow.
    """
    def __defaults__(self):
        
        # Initialise cryocooler properties as null values
        self.cooler_type                    =  ''
        self.rated_power                    =  0.0
        self.min_cryo_temp                  =  0.0 
        self.ambient_temp                   =  300.0
        self.inputs.cooling_power           = 0.0
        self.inputs.cryo_temp               = 0.0
        self.mass_properties.mass           = 0.0
        self.rated_power                    = 0.0
        self.outputs.input_power            = 0.0

    def energy_calc(self, conditions):

        """Calculate the instantaneous required energy input

        Assumptions:
        N/A

        Source:
        N/A

        Inputs:
        self.inputs
            cryo_temp       [K]
            cooling_power   [W]

        Outputs:
        self.outputs.
            input_power     [W]

        Properties Used:

        self.
            cooler_type 

    """ 
        output = cryocooler_model(self)
        self.outputs.input_power = output[0]

        return output[0]

    


## @ingroup Components-Energy-Cooling
# Cryocooler.py
# 
# Created:  Feb 2020,   K.Hamilton


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports
from SUAVE.Core import Data
from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Cooling.Cryocooler.Cooling.cryocooler_model import cryocooler_model 

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
        self.cooler_type        = 'GM'
        self.rated_power        =   0.0
        self.min_cryo_temp      =  77.0
        self.ambient_temp       = 300.0
        self.cooling_model      = cryocooler_model

        
    def energy_calc(self, cooling_power, cryo_temp, amb_temp):

        """This is .........................

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
        
        # Calculate the instantaneous required energy input
        output = self.cooling_model(self, cooling_power, cryo_temp, amb_temp)
        return output[0]
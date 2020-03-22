## @ingroup Analyses-Weights

# Weights_Electric_Lift_Cruise.py
#
# Created:  Aug, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from .Weights import Weights


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Weights
class Weights_Electric_Lift_Cruise(Weights):
    """ SUAVE.Analyses.Weights.Weights_Electric_Lift_Cruise()
    
    Assumptions:
    None
    
    Source:
    N/A
    
    Inputs:
    N/A
    
    Outputs:
    N/A
    
    Properties Used:
    N/A
    """

    def __defaults__(self):
        """Sets the default parameters for the weight analysis

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        N/A

        Outputs:
        N/A

        Properties Used:
        N/A
        """
        
        self.tag = 'weights_electric_stopped_rotor'
        
        self.vehicle  = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions=None):
        """Uses the weight buildup method to estimate vehicle weight

        Assumptions:
        Analysis has been assigned a vehicle.
        Weight method to be used is default Electric Stopped Rotor method.

        Source:
        N/A

        Inputs:
        Flight conditions, optionally

        Outputs:
        Weight breakdown of vehicle
        Vehicle object modified so as to include weight breakdown and empty operating weight

        Properties Used:
        Analysis-assigned vehicle
        """
        
        # unpack
        vehicle = self.vehicle
        empty   = SUAVE.Methods.Weights.Buildups.Electric_Lift_Cruise.empty

        
        # evaluate
        results = empty(vehicle)
        
        # storing weigth breakdown into vehicle
        vehicle.weight_breakdown = results 

        # updating empty weight
        vehicle.mass_properties.operating_empty = results.empty
              
        # done!
        return results
    
    
    def finalize(self):
        """Finalizes the results of the analysis

        Assumptions:
        Vehicle has been assigned mass properties

        Source:
        N/A

        Inputs:
        Analysis-assigned vehicle

        Outputs:
        Analysis object assigned vehicle mass properties

        Properties Used:
        Analysis-assigned vehicle mass properties
        """
        
        self.mass_properties = self.vehicle.mass_properties
        
        return
        

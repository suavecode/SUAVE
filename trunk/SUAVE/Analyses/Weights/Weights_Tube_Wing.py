## @ingroup Analyses-Weights
# Weights_Tube_Wing.py
#
# Created:  Apr 2017, Matthew Clarke
# Modified: Oct 2017, T. MacDonald

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
class Weights_Tube_Wing(Weights):
    """ This is class that evaluates the weight of Tube and Wing aircraft
    
    Assumptions:
        None

    Source:
        N/A

    Inputs:
        None
      
    Outputs:
        None
        
    Properties Used:
         N/A
    """

    def __defaults__(self):
        """This sets the default values and methods for the tube and wing 
        aircraft weight analysis.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """
        self.tag = 'weights_tube_wing'

        self.vehicle = Data()
        self.settings = Data()
        self.settings.weight_reduction_factors = Data()
        # Reduction factors are proportional (.1 is a 10% weight reduction)
        self.settings.weight_reduction_factors.main_wing = 0.
        self.settings.weight_reduction_factors.fuselage = 0.
        self.settings.weight_reduction_factors.empennage = 0.  # applied to horizontal and vertical stabilizers

    def evaluate(self, method="SUAVE", conditions=None):
        """Evaluate the weight analysis.
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        results

        Properties Used:
        N/A
        """
        # unpack
        vehicle = self.vehicle
        settings = self.settings
        results = SUAVE.Methods.Weights.Correlations.Common.empty_weight(vehicle, method_type=method)
        # if method == "Tube_Wing":
        #     results = SUAVE.Methods.Weights.Correlations.Tube_Wing.empty(vehicle, settings, conditions)
        # elif method == "FLOPS Simple":
        #     self.settings.complexity = "Simple"
        #     results = SUAVE.Methods.Weights.Correlations.FLOPS.empty(vehicle, settings)
        # elif method == "FLOPS Complex":
        #     self.settings.complexity = "Complex"
        #     results = SUAVE.Methods.Weights.Correlations.FLOPS.empty(vehicle, settings, conditions)
        # elif method == "New SUAVE":
        #     self.settings.complexity = "Complex"
        #     results = SUAVE.Methods.Weights.Correlations.Common.arbitrary(vehicle, settings,conditions,
        #                                                                   main_wing_calc_type="SUAVE" )
        # elif method == "Raymer":
        #     self.settings.complexity = "Complex"
        #     results = SUAVE.Methods.Weights.Correlations.Common.arbitrary(vehicle, settings,
        #                                                                  main_wing_calc_type="Raymer")

        # else:
        #     ValueError("This method has not been implemented")

        # storing weigth breakdown into vehicle
        vehicle.weight_breakdown = results

        # updating empty weight
        vehicle.mass_properties.operating_empty = results.empty

        # done!
        return results

    def finalize(self):
        """Finalize the weight analysis.
    
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """
        self.mass_properties = self.vehicle.mass_properties

        return

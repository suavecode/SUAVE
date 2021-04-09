## @ingroup Analyses-Costs
# Costs.py
#
# Created:  Sep 2016, T. Orra

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data
from SUAVE.Analyses import Analysis

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------
## @ingroup Analyses-Costs
class Costs(Analysis):
    """ This is the base class for cost analyses. It contains functions
    that are built into the default class.
    
    Assumptions:
    None
    
    Source:
    N/A
    """
    def __defaults__(self):
        """This sets the default values for the analysis to function.

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
        self.tag = 'costs'
        self.vehicle  = Data()

        # Default methods to be used
        self.settings = Data()
        self.settings.operating_costs_method = \
            SUAVE.Methods.Costs.Correlations.Operating_Costs.compute_operating_costs
        self.settings.industrial_costs_method = \
            SUAVE.Methods.Costs.Correlations.Industrial_Costs.compute_industrial_costs

    def evaluate(self,conditions=None):
        """This sets the default evaluation method for costs.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None used

        Outputs:
        None

        Properties Used:
        self.
          vehicle                           SUAVE vehicle passed to the functions below
          settings.industrial_costs_method  (function)
          settings.operating_costs_method   (function)
        """
        # unpack
        vehicle             = self.vehicle
        industrial_costs    = self.settings.industrial_costs_method
        operating_costs     = self.settings.operating_costs_method

        # evaluate
        results_manufacturing = industrial_costs(vehicle)
        results_operating     = operating_costs(vehicle)

       # done!
        return


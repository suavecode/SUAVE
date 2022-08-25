## @ingroup Analyses-Emission
# Emission.py
#
# Created:  May 2020, S. Karpuk

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
class Emission(Analysis):
    """ This is the base class for emission analyses. It contains functions
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
        self.tag = 'emission'
        self.vehicle  = Data()

        # Default methods to be used
        self.settings = Data()
        self.settings.emission_method = \
            SUAVE.Methods.Emission.compute_emission


    def evaluate(self,conditions=None):
        """This sets the default evaluation method for emission

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

        """
        # unpack
        vehicle             = self.vehicle
        mission             = self.mission
        emission            = SUAVE.Methods.Emission.Emission

        # evaluate
        results = emission(vehicle,mission)


        return results



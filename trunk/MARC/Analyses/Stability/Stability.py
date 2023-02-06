## @ingroup Analyses-Stability
# Stability.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from MARC.Core import Data
from MARC.Analyses import Analysis


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Stability
class Stability(Analysis):
    """ MARC.Analyses.Stability.Stability()
    """

    def __defaults__(self):
        """This sets the default values and methods for the analysis.

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
        self.tag    = 'stability'
        self.geometry = Data()
        self.settings = Data()

    def evaluate(self,conditions):
        """Evaluate the stability analysis.

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
        results = Data()

        return results


    def finalize(self):
        """Finalize the stability analysis.

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

        return



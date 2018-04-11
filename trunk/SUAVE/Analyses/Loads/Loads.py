## @ingroup Analyses-Loads
# Loads.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Analyses import Analysis


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

## @ingroup Analyses-Loads
class Loads(Analysis):
    """ SUAVE.Analyses.Loads.Loads()
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
        self.tag    = 'loads'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,condtitions):
        """Evaluate the Loads analysis.
        
                Assumptions:
                None
        
                Source:
                N/A
        
                Inputs:
                None
        
                Outputs:
                Results of the Loads Analysis
        
                Properties Used:
                N/A                
            """        
        return Data()
        
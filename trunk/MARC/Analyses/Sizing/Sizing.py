## @ingroup Analyses-Sizing
# Sizing.py
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

## @ingroup Analyses-Sizing
class Sizing(Analysis):
    """ MARC.Analyses.Sizing.Sizing()
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
        
        self.tag    = 'sizing'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions):
        """Evaluate the sizing analysis.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            Results of the Sizing Analysis
    
            Properties Used:
            N/A                
        """
        
        return Data()
    
    __call__ = evaluate
        
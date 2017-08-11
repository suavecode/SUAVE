## @ingroup Analyses-Structures
# Structures.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from SUAVE.Analyses import Analysis, Results


# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Structures(Analysis):
    """ SUAVE.Analyses.Structures.Structures()
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
        self.tag    = 'structures'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,conditions):
        """Evaluate the Structures analysis.
            
                Assumptions:
                None
            
                Source:
                N/A
            
                Inputs:
                None
            
                Outputs:
                Results of the Structures Analysis
            
                Properties Used:
                N/A                
            """
        return Results()
    
    __call__ = evaluate
        
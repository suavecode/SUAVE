## @ingroup Analyses-Geometry
# Geometry.py
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

## @ingroup Analyses-Geometry
class Geometry(Analysis):
    """ This class defines the geometry data structure of an aircraft in SUAVE
    
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
        self.tag    = 'geometry'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,condtitions):
        """Evaluate the Geometry analysis.
            
                Assumptions:
                None
            
                Source:
                N/A
            
                Inputs:
                None
            
                Outputs:
                Results of the Geometry Analysis
            
                Properties Used:
                N/A                
            """        
        return Results()
    
        
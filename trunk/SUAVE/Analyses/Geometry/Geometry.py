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
        self.tag    = 'geometry'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,condtitions):
        return Results()
    
        
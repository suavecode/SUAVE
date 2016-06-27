# Loads.py
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

class Loads(Analysis):
    """ SUAVE.Analyses.Loads.Loads()
    """
    def __defaults__(self):
        self.tag    = 'loads'
        self.features = Data()
        self.settings = Data()
        
        
    def evaluate(self,condtitions):
        return Results()
        
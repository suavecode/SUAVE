# Noise.py
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

class Noise(Analysis):
    """ SUAVE.Analyses.Aerodynamics.Aerodynamics()
    """
    def __defaults__(self):
        self.tag    = 'aerodynamics'
        
        self.geometry = Data()
        self.settings = Data()
        
        
    def evaluate(self,state):
        
        results = Data()
        
        return results
    
    def finalize(self):
        
        return  
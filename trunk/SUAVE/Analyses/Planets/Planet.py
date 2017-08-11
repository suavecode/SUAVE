## @ingroup Analyses-Planets
# Planet.py
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

## @ingroup Analyses-Planets
class Planet(Analysis):
    """ SUAVE.Analyses.Planet()
    """
    
    def __defaults__(self):
        
        """This sets the default values and methods for the analysis.
    
            Assumptions:
            Planet is Earth.
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            N/A
            """                  
        
        
        self.tag    = 'planet'
        self.features = Data()
        self.settings = Data()
        
        from SUAVE.Attributes.Planets.Earth import Earth
        self.features = Earth()
        
        
        
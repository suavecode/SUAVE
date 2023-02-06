## @ingroup Analyses-Planets
# Planet.py
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

## @ingroup Analyses-Planets
class Planet(Analysis):
    """ MARC.Analyses.Planet()
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
        
        from MARC.Attributes.Planets.Earth import Earth
        self.features = Earth()
        
        
        
# Markup.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data
from Aerodynamics import Aerodynamics
from SUAVE.Analyses import Process

# ----------------------------------------------------------------------
#  Analysis
# ----------------------------------------------------------------------

class Markup(Aerodynamics):
    """ SUAVE.Analyses.Aerodynamics.Markup()
    """
    def __defaults__(self):
        
        self.tag    = 'aerodynamics_markup'
        
        self.geometry = Data()
        self.settings = Data()
        
        self.process = Process()
        self.process.initialize = Process()
        self.process.compute = Process()
        
        
    def evaluate(self,state):
        
        settings = self.settings
        geometry = self.geometry
        
        results = self.process.compute(state,settings,geometry)
        
        return results
        
    def initialize(self):
        self.process.initialize(self)
    
        
        
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core import Data, Data_Exception, Data_Warning
from Aerodynamics import Aerodynamics
from SUAVE.Analyses import Process

# default Aero Results
from Results import Results

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
    
        
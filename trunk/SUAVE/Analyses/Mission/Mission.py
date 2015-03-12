""" Mission.py: Top-level mission class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data, Data_Exception
from SUAVE.Core import Container as ContainerBase

from SUAVE.Methods import Missions as Methods

import Segments

# ----------------------------------------------------------------------
#   Class
# ----------------------------------------------------------------------

class Mission(Segments.Simple.Container):
    """ Mission.py: Top-level mission class """
    
    def __defaults__(self):
        self.tag = 'mission'
        
        # see Segments.Simple.Container
        
    def finalize(self):
        pass
    
    

# ----------------------------------------------------------------------
#   Cotnainer Class
# ----------------------------------------------------------------------

class Container(ContainerBase):
    
    def evaluate(self,state=None):
        results = SUAVE.Analyses.Results()
        
        for key,mission in self.items():
            result = mission.evaluate(state)
            results[key] = result
            
        return results
    
    def finalize(self):
        pass

# Link container
Mission.Container = Container

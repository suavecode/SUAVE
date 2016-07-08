# Mission.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

""" Mission.py: Top-level mission class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import ContainerOrdered as ContainerBase

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
#   Container Class
# ----------------------------------------------------------------------

class Container(ContainerBase):
    
    def evaluate(self,state=None):
        #results = SUAVE.Analyses.Results()
        result = SUAVE.Core.DataOrdered()
        
        for key,mission in self.items():
            result = mission.evaluate(state)
            results[key] = result
            
        return results
    
    def finalize(self):
        pass

# Link container
Mission.Container = Container

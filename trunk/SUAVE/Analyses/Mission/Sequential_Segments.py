""" Mission.py: Top-level mission class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Data, Data_Exception
from SUAVE.Core import Container as ContainerBase

from SUAVE.Methods import Missions as Methods

from Mission import Mission

# ----------------------------------------------------------------------
#   Class
# ----------------------------------------------------------------------

class Sequential_Segments(Mission):
    """ Solves each segment one at time """
    
    def __defaults__(self):
        
        self.tag = 'mission'
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        self.process.initialize = Methods.Segments.Common.Sub_Segments.expand_sub_segments

        # --------------------------------------------------------------
        #   Converge
        # --------------------------------------------------------------
        self.process.converge = Methods.Segments.Common.Sub_Segments.sequential_sub_segments
        
        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------        
        del self.process.iterate

        # --------------------------------------------------------------
        #   Finalize
        # --------------------------------------------------------------        
        self.process.finalize.sub_segments = Methods.Segments.Common.Sub_Segments.finalize_sub_segments
        
        return
    
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

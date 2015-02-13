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

# Link container
Mission.Container = Container

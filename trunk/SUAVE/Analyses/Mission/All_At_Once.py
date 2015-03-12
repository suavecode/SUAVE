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

class All_At_Once(Mission):
    """ Mission.py: Top-level mission class """
    
    def __defaults__(self):
        
        self.tag = 'mission'
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        self.process.initialize.expand_state        = Methods.Segments.expand_state
        self.process.initialize.expand_sub_segments = Methods.Segments.Common.Sub_Segments.expand_sub_segments

        # --------------------------------------------------------------
        #   Converge
        # --------------------------------------------------------------
        self.process.converge.converge_root         = Methods.Segments.converge_root
        
        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------        
        self.process.iterate.sub_segments           = Methods.Segments.Common.Sub_Segments.update_sub_segments

        # --------------------------------------------------------------
        #   Finalize
        # --------------------------------------------------------------        
        self.process.finalize.sub_segments          = Methods.Segments.Common.Sub_Segments.finalize_sub_segments
        
    def finalize(self):
        pass
    
    
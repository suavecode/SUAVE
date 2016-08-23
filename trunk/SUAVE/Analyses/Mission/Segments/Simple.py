# Simple.py
#
# Created:  
# Modified: Feb 2016, Andrew Wendorff

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses import Process
from SUAVE.Analyses.Mission.Segments import Segment
from SUAVE.Analyses.Mission.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Simple(Segment):
    
    def __defaults__(self):
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Basic() )
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize - before iteration
        # --------------------------------------------------------------
        initialize = self.process.initialize
        
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = None
        
        # --------------------------------------------------------------
        #   Converge - starts iteration
        # --------------------------------------------------------------
        converge = self.process.converge
        
        converge.converge_root             = Methods.converge_root

        # --------------------------------------------------------------
        #   Iterate - this is iterated
        # --------------------------------------------------------------
        iterate = self.process.iterate
                
        # Update Initials
        iterate.initials = Process()
        iterate.initials.time              = Methods.Common.Frames.initialize_time
        
        # Unpack Unknowns
        iterate.unknowns = Process()
        iterate.unknowns.mission           = None
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = Methods.Common.Numerics.update_differentials_time
        
        # Solve Residuals
        iterate.residuals = Process()

        # --------------------------------------------------------------
        #   Finalize - after iteration
        # --------------------------------------------------------------
        finalize = self.process.finalize
        
        finalize.post_process = Process()
        
        
        return


# ----------------------------------------------------------------------
#  Container
# ----------------------------------------------------------------------

class Container(Segment.Container):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        self.process.initialize.expand_state        = Methods.expand_state
        self.process.initialize.expand_sub_segments = Methods.Common.Sub_Segments.expand_sub_segments

        # --------------------------------------------------------------
        #   Converge
        # --------------------------------------------------------------
        self.process.converge.converge_root         = Methods.converge_root
        
        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------        
        self.process.iterate.sub_segments           = Methods.Common.Sub_Segments.update_sub_segments

        # --------------------------------------------------------------
        #   Finalize
        # --------------------------------------------------------------        
        self.process.finalize.sub_segments          = Methods.Common.Sub_Segments.finalize_sub_segments
    
        
        return
        
    
Simple.Container = Container
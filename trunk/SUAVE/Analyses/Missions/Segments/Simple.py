
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses import Process
from SUAVE.Analyses.Missions.Segments import Segment
from SUAVE.Analyses.Missions.Segments import Conditions

from SUAVE.Methods.Missions import Segments as Methods


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Simple(Segment):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        # self.example = 1.0
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Basic() )
        
        # initials and unknowns, Example...
        ##ones_row = self.state.ones_row
        ##self.state.unknowns.throttle   = ones_row(1) * 0.5
        ##self.state.unknowns.body_angle = ones_row(1) * 0.0
        ##self.state.residuals.forces    = ones_row(2) * 0.0
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.expand_state            = Methods.expand_state
        initialize.differentials           = Methods.Common.Numerics.initialize_differentials_dimensionless
        initialize.conditions              = None
        
        # --------------------------------------------------------------
        #   Converge
        # --------------------------------------------------------------
        converge = self.process.converge
        converge.converge_root             = Methods.converge_root

        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------
        iterate = self.process.iterate
                
        # Update Initials
        iterate.initials = Process()
        iterate.initials.time              = Methods.Common.Frames.initialize_time
        
        # Unpack Unknowns
        iterate.unpack_unknowns            = None        
        
        # Update Conditions
        iterate.conditions = Process()
        iterate.conditions.differentials   = Methods.Common.Numerics.update_differentials_time
        
        # Solve Residuals
        iterate.residuals = Process()

        # --------------------------------------------------------------
        #   Finalize
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
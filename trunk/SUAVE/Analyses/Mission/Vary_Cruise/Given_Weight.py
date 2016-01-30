""" Mission.py: Top-level mission class """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods import Missions as Methods

from SUAVE.Analyses.Mission import All_At_Once

# ----------------------------------------------------------------------
#   Class
# ----------------------------------------------------------------------

class Given_Weight(All_At_Once):
    """ Mission.py: Top-level mission class """
    
    def __defaults__(self):
        
        self.tag = 'vary_cruise_given_weight'
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.cruise_tag  = 'cruise'
        self.target_landing_weight = 1000.0
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # initials and unknowns, on top of segment initials and unknowns
        self.state.unknowns.cruise_distance  = 1000.0
        self.state.residuals.landing_weight  = 0.0
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        self.process.initialize.expand_state        = Methods.Segments.expand_state
        self.process.initialize.expand_sub_segments = Methods.Segments.Common.Sub_Segments.expand_sub_segments
        self.process.initialize.cruise_distance     = Methods.Segments.Cruise.Variable_Cruise_Distance.initialize_cruise_distance
        
        # --------------------------------------------------------------
        #   Converge
        # --------------------------------------------------------------
        self.process.converge.converge_root         = Methods.Segments.converge_root
        
        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------        
        iterate = self.process.iterate
        iterate.clear()        
        
        # unpack the unknown
        iterate.unpack_distance              = Methods.Segments.Cruise.Variable_Cruise_Distance.unknown_cruise_distance
        
        # Run the Segments
        iterate.sub_segments                 = Methods.Segments.Common.Sub_Segments.update_sub_segments
        
        # Solve Residuals
        self.process.iterate.residual_weight = Methods.Segments.Cruise.Variable_Cruise_Distance.residual_landing_weight
        
        
        # --------------------------------------------------------------
        #   Finalize
        # --------------------------------------------------------------        
        self.process.finalize.sub_segments          = Methods.Segments.Common.Sub_Segments.finalize_sub_segments
        

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Analyses.New_Segment import Aerodynamic
from SUAVE.Analyses.New_Segment import Conditions

from SUAVE.Methods.Missions import Segments as Methods

# Units
from SUAVE.Core import Units


# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Cruise(Aerodynamic):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude  = 10. * Units.km
        self.air_speed = 10. * Units['km/hr']
        self.distance  = 10. * Units.km
        
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # conditions
        self.state.conditions.update( Conditions.Aerodynamics() )
        
        # initials and unknowns
        ones_row = self.state.ones_row
        self.state.unknowns.throttle   = ones_row(1) * 0.5
        self.state.unknowns.body_angle = ones_row(1) * 0.0
        self.state.residuals.forces    = ones_row(2) * 0.0
        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        
        # --------------------------------------------------------------
        #   Initialize
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.conditions              = Methods.Cruise.Common.initialize_conditions


        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------
        iterate = self.process.iterate
        
        # Unpack Unknowns
        iterate.unpack_unknowns            = Methods.Cruise.Common.unpack_unknowns
                        
        # Solve Residuals
        iterate.residuals.total_forces     = Methods.Cruise.Common.residual_total_forces

        
        return


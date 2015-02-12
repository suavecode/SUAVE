
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

class Constant_Speed_Constant_Rate(Aerodynamic):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start = None # Optional
        self.altitude_end   = 10. * Units.km
        self.climb_rate     = 3.  * Units.m / Units.s
        self.air_speed      = 100 * Units.m / Units.s
        self.battery_energy = 0.0
        self.latitude       = 0.0
        self.longitude      = 0.0
        
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
        initialize.conditions = Methods.Climb.Constant_Speed_Constant_Rate.initialize_conditions
        initialize.differentials_altitude = Methods.Climb.Common.update_differentials_altitude

        # --------------------------------------------------------------
        #   Iterate
        # --------------------------------------------------------------
        iterate = self.process.iterate
        del self.process.iterate.conditions.differentials
        
        # Unpack Unknowns
        iterate.unpack_unknowns            = Methods.Climb.Common.unpack_unknowns
                        
        # Solve Residuals
        iterate.residuals.total_forces     = Methods.Climb.Common.residual_total_forces

        
        return


# Constant_Dynamic_Pressure_Constant_Rate.py
#
# Created:  Jun 2017, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Methods.Missions import Segments as Methods

from Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

class Constant_Dynamic_Pressure_Constant_Angle(Unknown_Throttle):
    
    def __defaults__(self):
        
        # --------------------------------------------------------------
        #   User inputs
        # --------------------------------------------------------------
        self.altitude_start   = None # Optional
        self.altitude_end     = 10.  * Units.km
        self.climb_angle      = 3.   * Units.degrees
        self.dynamic_pressure = 1600 * Units.pascals
        
        # --------------------------------------------------------------
        #   State
        # --------------------------------------------------------------
        
        # initials and unknowns
        ones_row = self.state.ones_row        
        self.state.unknowns.altitudes  = ones_row(1) * 0.0
        self.state.residuals.forces    = ones_row(3) * 0.0        
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        initialize.conditions = Methods.Climb.Constant_Dynamic_Pressure_Constant_Angle.initialize_conditions_unpack_unknowns
        
        # Unpack Unknowns
        iterate = self.process.iterate
        iterate.unknowns.mission           = Methods.Climb.Constant_Dynamic_Pressure_Constant_Angle.initialize_conditions_unpack_unknowns
        
        iterate.conditions.differentials   = Methods.Climb.Optimized.update_differentials
        
        # Solve Residuals
        iterate.residuals.total_forces     = Methods.Climb.Constant_Dynamic_Pressure_Constant_Angle.residual_total_forces        
    
        return
       
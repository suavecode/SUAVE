## @ingroup Analyses-Mission-Segments-Climb
# Constant_Dynamic_Pressure_Constant_Angle.py
#
# Created:  Jun 2017, E. Botero
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUAVE imports
from SUAVE.Methods.Missions import Segments as Methods

from .Unknown_Throttle import Unknown_Throttle

# Units
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Climb
class Constant_Dynamic_Pressure_Constant_Angle(Unknown_Throttle):
    """ Climb at a constant dynamic pressure at a constant angle.
        This segment takes longer to solve than most because it has extra unknowns and residuals
    
        Assumptions:
        None
        
        Source:
        None
    """       
    
    def __defaults__(self):
        """ This sets the default solver flow. Anything in here can be modified after initializing a segment.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """          
        
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
       
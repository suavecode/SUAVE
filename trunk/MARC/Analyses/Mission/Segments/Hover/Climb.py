## @ingroup Analyses-Mission-Segments-Hover
# Climb.py
# 
# Created:  Jan 2016, E. Botero
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC imports
import MARC
from MARC.Analyses.Mission.Segments import Aerodynamic
from MARC.Analyses.Mission.Segments import Conditions

from MARC.Methods.Missions import Segments as Methods

from MARC.Analyses import Process
from .Hover import Hover

# Units
from MARC.Core import Units

# ----------------------------------------------------------------------
#  Segment
# ----------------------------------------------------------------------

## @ingroup Analyses-Mission-Segments-Hover
class Climb(Hover):
    """ A vertically climbing hover for VTOL aircraft. Although the vehicle moves, no aerodynamic drag and lift are used.
    
        Assumptions:
        Your vehicle creates a negligible drag and lift force during a vertical climb.
        
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
        self.altitude_start    = None # Optional
        self.altitude_end      = 1. * Units.km
        self.climb_rate        = 1.  * Units.m / Units.s
        self.true_course_angle = 0.0 * Units.degrees 
        
        # --------------------------------------------------------------
        #   The Solving Process
        # --------------------------------------------------------------
        initialize = self.process.initialize
        iterate    = self.process.iterate
        
        initialize.conditions = Methods.Hover.Climb.initialize_conditions
    
        return
       
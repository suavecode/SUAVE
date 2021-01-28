## @ingroup Components-Wings-Control_Surfaces
# Control_Surface.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald
#           Jun 2017, M. Clarke
#           Aug 2019, M. Clarke
# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core       import Data , Units
from SUAVE.Components import Physical_Component
from SUAVE.Components import Lofted_Body
import numpy as np 
    
# ------------------------------------------------------------
#  Control Surfaces
# ------------------------------------------------------------

## @ingroup Components-Wings-Control_Surfaces
class Control_Surface(Physical_Component):
    def __defaults__(self):
        """This sets the default values of control surfaces defined in SUAVE. 
        sign_duplicate: 1.0 or -1.0 - the sign of the duplicate control on the mirror wing.
        
        Use 1.0 for a mirrored control surface, like an elevator. Use -1.0 for an aileron.
        
        The span fraction is given by the array shown below:  
        [abs. % span location at beginning of crtl surf, abs. % span location at end  of crtl surf]
        
        The function argumentis a string that defines the function of a control surface. Options
        are 'elevator','rudder','flap', 'aileron' and 'slat'
        
        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        N/A
        """         

        self.tag                   = 'control_surface' 
        self.span                  = 0.0
        self.span_fraction_start   = 0.0
        self.span_fraction_end     = 0.0
        self.chord_fraction        = 0.0  
        self.hinge_fraction        = 1.0
        self.deflection            = 0.0  
        self.configuration_type    = 'single_slotted'
        self.gain                  = 1.0

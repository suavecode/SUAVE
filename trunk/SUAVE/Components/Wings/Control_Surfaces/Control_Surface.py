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
        
        - span: the span of the control surface in meters        
        - span_fraction_start: % span of the wing where the control surface starts        
        - span_fraction_start: % span of the wing where the control surface starts
        
        - hinge_fraction: number between 0.0 and 1.0. This corresponds to the location of the 
            hingeline, where 0 and 1 correspond to the leading and trailing edges, respectively, 
            of the CONTROL SURFACE (NOT the wing).
        - chord_fraction: number between 0.0 and 1.0 describing the fraction of the wing's chord
            that is 'cut' out by the control surface
        
        - sign_duplicate: 1.0 or -1.0 - the sign of the duplicate control on the mirror wing.        
            Use 1.0 for a mirrored control surface, like an elevator. Use -1.0 for an aileron.
        - deflection: the angle the control surface is deflected. 
        - configuration_type: the kind of control surface (e.g. single_slotted)
        
        Assumptions:
        - for chord_fraction, Slats are always cut out from the leading edge and everything else
            os cut out from the trailing edge.

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
        
        self.hinge_fraction        = 0.0
        self.chord_fraction        = 0.0
        
        self.sign_duplicate        = 1.0
        self.deflection            = 0.0  
        self.configuration_type    = 'single_slotted'
        
        self.gain                  = 1.0 #deflection multiplier used only for AVL

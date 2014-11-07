# extend_to_ref_area.py
#
# Created: March 2014, Tim Momose
# IN PROGRESS

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from SUAVE.Components.Wings.Wing import Wing
from SUAVE.Attributes import Units as Units
from SUAVE.Structure import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def extend_to_ref_area(surface,height_above_centerline):
    """ref_surface = SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area(wing,) 
        This method takes inputs describing the exposed portion of a trapezoidal
        aerodynamic surface and calculates the dimensions of a corresponding 
        aerodynamic surface that extends all the way to the fuselage centerline.
        Particularly used to get the vertical tail reference area for lateral 
        stability calculations when the dimensions of the exposed tail are known.
        
        Inputs:
            surface - a SUAVE Wing object with the fields:
                span - span (height for a vertical tail) of the exposed surface
                [meters]
                sweep - leading edge sweep of the aerodynamic surface [radians]
                root_chord - chord length at the junction between the tail and 
                the fuselage [meters]
                tip_chord - chord length at the tip of the aerodynamic surface
                [meters]
                
            height_above_centerline - the displacement from the fuselage
            centerline to the exposed area's physical root chordline [meters]
    
        Outputs:
            ref_surface - a data dictionary with the fields:
                ref_span - The span/height measured from the fuselage centerline
                [meters]
                ref_area - The area of the extended trapezoidal surface 
                [meters**2]
                ref_aspect_ratio - The aspect ratio of the extended surface 
                [meters]
                ref_root_chord - The chord of the extended trapezoidal surface 
                where it meets the fuselage centerline [meters]
                root_LE_change - The change in the leading edge position of the
                surface compared to the smaller surface that only extended to the
                fuselage surface. This value is negative for sweptback surfaces
                [meters]
                
        Assumptions:
            Assumes a simple trapezoidal half-wing shape.
    """             
    # Unpack inputs
    b1        = surface.spans.exposed
    c_t       = surface.chords.tip
    c_r1      = surface.chords.fuselage_intersect
    Lambda    = surface.sweep
    dh_center = height_above_centerline
#    print 'b: {}; dh: {}'.format(b1,dh_center)
    
    #Compute reference area dimensions
    b      = b1+dh_center
    c_root = c_t + (b/b1)*(c_r1-c_t)
    S      = 0.5*b*(c_root+c_t)
    dx_LE  = -dh_center*np.tan(Lambda)
    AR     = b**2/S
    
    surface.spans.projected   = b
    surface.areas.reference   = S
    surface.aspect_ratio      = AR
    surface.chords.root       = c_root
    surface.root_LE_change    = dx_LE
    
    return surface
# extend_to_ref_area.py
#
# Created: March 2014, Tim Momose
# IN PROGRESS

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np
from copy import deepcopy
from SUAVE.Components.Wings.Wing import Wing
from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

def extend_to_ref_area(surface):
    """ref_surface = SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area(wing,)
        This method takes inputs describing the exposed portion of a trapezoidal
        aerodynamic surface and calculates the dimensions of a corresponding
        aerodynamic surface that extends all the way to the fuselage centerline.
        Particularly used to get the vertical tail reference area for lateral
        stability calculations when the dimensions of the exposed tail are known.

        Inputs:
            surface - a SUAVE Wing object with the fields:
                spans.projected - projected span (height for a vertical tail) of
                 the exposed surface [meters]
                sweep - leading edge sweep of the aerodynamic surface [radians]
                chords.root - chord length at the junction between the tail and
                 the fuselage [meters]
                chords.tip - chord length at the tip of the aerodynamic surface
                 [meters]
                symmetric - Is the wing symmetric across the fuselage centerline?
                exposed_root_chord_offset - the displacement from the fuselage
                 centerline to the exposed area's physical root chordline [meters]

        Outputs:
            ref_surface - a data dictionary with the fields:
                spans.projected - The span/height measured from the fuselage centerline
                [meters]
                area.reference - The area of the extended trapezoidal surface
                [meters**2]
                aspect_ratio - The aspect ratio of the extended surface
                [meters]
                chords.root - The chord of the extended trapezoidal surface
                where it meets the fuselage centerline [meters]
                root_LE_change - The change in the leading edge position of the
                surface compared to the smaller surface that only extended to the
                fuselage surface. This value is negative for sweptback surfaces
                [meters]

        Assumptions:
            Assumes a simple trapezoidal half-wing shape.
    """
    # Unpack inputs
    symm      = surface.symmetric
    try:
        b1 = surface.spans.exposed * 0.5 * (2 - symm)
    except AttributeError:
        b1 = surface.spans.projected * 0.5 * (2 - symm)
    c_t       = surface.chords.tip
    c_r1      = surface.chords.root
    Lambda    = surface.sweep
    dh_center = surface.exposed_root_chord_offset
#    print 'b: {}; dh: {}'.format(b1,dh_center)

    #Compute reference area dimensions
    b      = b1+dh_center
    c_root = c_t + (b/b1)*(c_r1-c_t)
    S      = 0.5*b*(c_root+c_t)
    dx_LE  = -dh_center*np.tan(Lambda)
    AR     = b**2/S

    ref_surface = deepcopy(surface)
    ref_surface.spans.projected   = b * (1 + symm)
    ref_surface.areas.reference   = S * (1 + symm)
    ref_surface.aspect_ratio      = AR * (1 + symm)
    ref_surface.chords.root       = c_root
    ref_surface.root_LE_change    = dx_LE
    ref_surface.origin[0]         = ref_surface.origin[0] + dx_LE

    return ref_surface

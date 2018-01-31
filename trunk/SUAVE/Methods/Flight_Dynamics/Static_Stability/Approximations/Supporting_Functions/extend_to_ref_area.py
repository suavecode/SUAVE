## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Supporting_Functions
# extend_to_ref_area.py
#
# Created:  Mar 2014, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

#SUAVE Imports 
import numpy as np
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Method
# ----------------------------------------------------------------------

## @ingroup Methods-Flight_Dynamics-Static_Stability-Approximations-Supporting_Functions
def extend_to_ref_area(surface):
    """ This method takes inputs describing the exposed portion of a trapezoidal
    aerodynamic surface and calculates the dimensions of a corresponding
    aerodynamic surface that extends all the way to the fuselage centerline.
    Particularly used to get the vertical tail reference area for lateral
    stability calculations when the dimensions of the exposed tail are known.


    Assumptions:
        Assumes a simple trapezoidal half-wing shape.

    Source:
        Unknown
    
    Inputs:
        surface - a SUAVE Wing object with the fields:
            spans.projected - projected span (height for a vertical tail) of
             the exposed surface                                                     [meters]
            sweep - leading edge sweep of the aerodynamic surface                    [radians]
            chords.root - chord length at the junction between the tail and
             the fuselage                                                            [meters]
            chords.tip - chord length at the tip of the aerodynamic surface          [meters]                                                                                   
            symmetric - Is the wing symmetric across the fuselage centerline?
            exposed_root_chord_offset - the displacement from the fuselage
             centerline to the exposed area's physical root chordline                [meters]

    Outputs:
        ref_surface - a data dictionary with the fields:
            spans.projected - The span/height measured from the fuselage centerline  [meters]                                                                                                
            area.reference - The area of the extended trapezoidal surface            [meters**2]                                                                                                
            aspect_ratio - The aspect ratio of the extended surface                  [meters]                                                                                               
            chords.root - The chord of the extended trapezoidal surface
            where it meets the fuselage centerline                                   [meters]
            root_LE_change - The change in the leading edge position of the
            surface compared to the smaller surface that only extended to the
            fuselage surface. This value is negative for sweptback surfaces          [meters]
            
    Properties Used:
         N/A        
    """
    # Unpack inputs
    symm      = surface.symmetric
    try:
        b1 = surface.spans.exposed * 0.5 * (2 - symm)
    except AttributeError:
        b1 = surface.spans.projected * 0.5 * (2 - symm)
    c_t       = surface.chords.tip
    c_r1      = surface.chords.root
    Lambda    = surface.sweeps.quarter_chord
    dh_center = surface.exposed_root_chord_offset

    #Compute reference area dimensions
    b      = b1+dh_center
    c_root = c_t + (b/b1)*(c_r1-c_t)
    S      = 0.5*b*(c_root+c_t)
    dx_LE  = -dh_center*np.tan(Lambda)
    AR     = b**2/S

    ref_surface = surface
    surface.extended = Data()
    surface.extended.spans  = Data()
    surface.extended.areas  = Data()
    surface.extended.chords = Data()
    ref_surface.extended.origin            = np.array(surface.origin) * 1.
    ref_surface.extended.spans.projected   = b * (1 + symm)
    ref_surface.extended.areas.reference   = S * (1 + symm)
    ref_surface.extended.aspect_ratio      = AR * (1 + symm)
    ref_surface.extended.chords.root       = c_root
    ref_surface.extended.root_LE_change    = dx_LE
    ref_surface.extended.origin[0]         = ref_surface.origin[0] + dx_LE

    return ref_surface

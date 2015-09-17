# Geoemtry.py
#

""" SUAVE Methods for Geoemtry Generation
"""

# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Core  import Data


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def wing_planform(wing):
    """ err = SUAVE.Methods.Geometry.wing_planform(Wing)
    
        basic wing planform calculation
        
        Assumptions:
            trapezoidal wing
            no leading/trailing edge extensions
            
        Inputs:
            Wing.sref
            Wing.ar
            Wing.taper
            Wing.sweep
            
        Outputs:
            Wing.chord_root
            Wing.chord_tip
            Wing.chord_mac
            Wing.area_wetted
            Wing.span
        
    """
    
    # unpack
    sref  = wing.areas.reference
    taper = wing.taper
    sweep = wing.sweep
    ar    = wing.aspect_ratio
    t_c_w = wing.thickness_to_chord
    
    # calculate
    span = sqrt(ar*sref)
    chord_root = 2*sref/span/(1+taper)
    chord_tip  = taper * chord_root
    
    swet = 2.*span/2.*(chord_root+chord_tip) *  (1.0 + 0.2*t_c_w)

    mac = 2./3.*( chord_root+chord_tip - chord_root*chord_tip/(chord_root+chord_tip) )
    
    # update
    wing.chords.root                = chord_root
    wing.chords.tip                 = chord_tip
    wing.chords.mean_aerodynamic    = mac
    wing.areas.wetted               = swet
    wing.spans.projected            = span
    
    return wing

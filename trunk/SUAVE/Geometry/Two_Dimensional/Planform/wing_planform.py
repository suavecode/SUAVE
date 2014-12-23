# Geoemtry.py
#

""" SUAVE Methods for Geometry Generation
"""

# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy
from math import pi, sqrt
from SUAVE.Structure  import Data
#from SUAVE.Attributes import Constants
from SUAVE.Geometry.Two_Dimensional.Planform.CrankedPlanform import CrankedPlanform


# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def wing_planform(wing):
    """ err = SUAVE.Geometry.wing_planform(Wing)
    
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
    sref = wing.areas.reference
    taper = wing.taper
    sweep = wing.sweep
    ar = wing.aspect_ratio
    thickness_to_chord = wing.thickness_to_chord
    span_ratio_fuselage = wing.span_ratios.fuselage

    # compute wing planform geometry
    wpc = CrankedPlanform(sref, ar, sweep, taper,
                              thickness_to_chord, span_ratio_fuselage)

    # set the wing origin
    wpc.wing_origin(wing.origin)

    # compute
    wpc.update()

    # update
    wing.chords.root = wpc.chord_root
    wing.chords.tip = wpc.chord_tip
    wing.chords.mean_aerodynamic = wpc.mean_aerodynamic_chord
    wing.chords.mean_aerodynamic_exposed = wpc.mean_aerodynamic_chord_exposed
    wing.chords.mean_geometric = wpc.mean_geometric_chord

    wing.aerodynamic_center = wpc.aerodynamic_center

    wing.areas.wetted = wpc.area_wetted
    wing.areas.gross = wpc.area_gross
    wing.areas.exposed = wpc.area_exposed

    wing.spans.projected = wpc.span

    return wing

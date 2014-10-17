# Geoemtry.py
#

""" SUAVE Methods for Geoemtry Generation
"""

# TODO:
# object placement, wing location
# tail: placed at end of fuselage, or pull from volume
# engines: number of engines, position by 757

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import numpy
from math import pi, sqrt
from SUAVE.Structure import Data
from SUAVE.Geometry.Two_Dimensional.Planform.wing_planform import wing_planform
#from SUAVE.Attributes import Constants

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------


def main_wing_planform(Wing):
    """ err = SUAVE.Geometry.main_wing_planform(Wing)
        
        main wing planform
        
        Assumptions:
            cranked wing with leading and trailing edge extensions
        
        Inputs:
            Wing.sref
            Wing.ar
            Wing.taper
            Wing.sweep
            Wing.span
            Wing.lex
            Wing.tex
            Wing.span_chordext
    
        Outputs:
            Wing.chord_root
            Wing.chord_tip
            Wing.chord_mid
            Wing.chord_mac
            Wing.area_wetted
            Wing.span
    
    """

    # unpack
    span = Wing.span
    lex = Wing.lex
    tex = Wing.tex
    span_chordext = Wing.span_chordext

    # run basic wing planform
    # mac assumed on trapezoidal reference wing
    err = wing_planform(Wing)

    # unpack more
    chord_root = Wing.chords.root
    chord_tip = Wing.chords.tip

    # calculate

    # this is the trazoidal chord at midwing, naming is confusing
    chord_mid_trap = chord_root + span_chordext * (chord_tip - chord_root)

    swet = 2 * span / 2 * (span_chordext * (chord_root + lex + tex + chord_mid_trap) +
                           (1 - span_chordext) * (chord_mid_trap + chord_tip))

    # update
    Wing.chord_mid_trap = chord_mid_trap
    Wing.swet = swet

    return 0
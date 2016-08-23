# main_wing_planform.py
#
# Created:  Mar 2013, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Geometry.Two_Dimensional.Planform.wing_planform import wing_planform

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def main_wing_planform(Wing):
    """ err = SUAVE.Methods.Geometry.main_wing_planform(Wing)
        
        main wing planform
        
        Assumptions:
            cranked wing with leading and trailing edge extensions
        
        Inputs:
            Wing.sref
            Wing.ar
            Wing.taper
            Wing.sweeps.quarter_chord
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
    span          = Wing.span
    lex           = Wing.lex
    tex           = Wing.tex
    span_chordext = Wing.span_chordext    
    
    # run basic wing planform
    # mac assumed on trapezoidal reference wing
    err = wing_planform(Wing)
    
    # unpack more
    chord_root    = Wing.chord_root
    chord_tip     = Wing.chord_tip
    
    # calculate
    chord_mid = chord_root + span_chordext*(chord_tip-chord_root)
    
    swet = 2*span/2*(span_chordext*(chord_root+lex+tex + chord_mid) +
                     (1-span_chordext)*(chord_mid+chord_tip))    
    
    # update
    Wing.chord_mid = chord_mid
    Wing.swet      = swet

    return 0
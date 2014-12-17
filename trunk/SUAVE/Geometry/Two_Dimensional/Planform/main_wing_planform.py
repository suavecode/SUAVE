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

import numpy as np
from scipy import interpolate
from math import pi, sqrt
from SUAVE.Structure  import Data
from SUAVE.Geometry.Two_Dimensional.Planform.wing_planform import wing_planform
#from SUAVE.Attributes import Constants
from AeroSurfacePlanformGeometry import AeroSurfacePlanformGeometry

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
    span          = Wing.span
    lex           = Wing.lex
    tex           = Wing.tex
    span_ratio_break = Wing.span_chordext
    span_ratio_fuselage = Wing.span_ratio_fuselage
    thickness_to_chord = Wing.thickness_to_chord
    span_ratio_flap_outer = Wing.span_ratio_flap_outer

    # run basic wing planform
    # mac assumed on trapezoidal reference wing
    err = wing_planform(Wing)

    semi_span = span/2.

    # non-dimensional span eta = y/(b/2)
    eta_node = np.array([0, span_ratio_break, 1])
    y_node = eta_node*semi_span

    # unpack more
    chord_root_trap = Wing.chords.root
    chord_tip = Wing.chords.tip
    chord_break = chord_root_trap + span_ratio_break*(chord_tip-chord_root_trap)

    # trapzoidal wing definition points
    c_node_trap = np.array([chord_root_trap, chord_break, chord_tip])

    # calculate
    chord_root = chord_root_trap+lex+tex

    # chord of wing definition nodes
    c_node = np.array([chord_root, chord_break, chord_tip])

    # compute the wing surface
    planform_wing = AeroSurfacePlanformGeometry(c_node, y_node)
    planform_wing.update()

    # build a wing chord interpolant that can be reused
    Wing.chord_from_eta = planform_wing.get_chord_interpolator()

    # assume that wing sweep refers to the QC sweep
    sweep_qc = Wing.sweep

    # compute x leading edge so that we can compute xac
    # TODO: current implementation does not account for LEX
    # TODO: make this genetic
    dx_qc = planform_wing.dy*np.tan(np.radians(sweep_qc))
    x_qc_root_trap = chord_root_trap*0.25
    x_qc_break_trap = x_qc_root_trap + dx_qc[0]
    x_qc_tip_trap = x_qc_break_trap + dx_qc[1]
    x_qc_trap_node = [x_qc_root_trap, x_qc_break_trap, x_qc_tip_trap]

    # get the leading edge x coordinates
    x_le_trap_node = x_qc_trap_node - c_node_trap/4.
    x_le_node = x_le_trap_node  # TODO: this is only true for LEX=0

    # compute the aerodynamic center
    x_ac_local = planform_wing.calc_aerodynamic_center(x_le_node)

    # TODO: this is a limitation, should be addressed in the future using a sort/set operation
    assert(span_ratio_break > span_ratio_fuselage)

    # get the fuselage-wing intersection chord
    chord_fuse_intersection = Wing.chord_from_eta(span_ratio_fuselage)
    c_node_exposed = np.array([chord_fuse_intersection, chord_break, chord_tip])
    eta_node_exposed = np.array([span_ratio_fuselage, span_ratio_break, 1])
    y_node_exposed = eta_node_exposed*semi_span
    planform_exposed = AeroSurfacePlanformGeometry(c_node_exposed, y_node_exposed)
    planform_exposed.update()

    # TODO: this is a limitation, should be addressed in the future using a sort/set operation
    assert(span_ratio_flap_outer > span_ratio_break)
    assert(span_ratio_flap_outer < semi_span)

    chord_flap_outer = Wing.chord_from_eta(span_ratio_flap_outer)
    c_node_flapped = np.array([chord_fuse_intersection, chord_break, chord_flap_outer])
    eta_node_flapped = np.array([span_ratio_fuselage, span_ratio_break, span_ratio_flap_outer])
    y_node_flapped = eta_node_flapped*semi_span
    planform_flapped = AeroSurfacePlanformGeometry(c_node_flapped, y_node_flapped)
    planform_flapped.update()

    # update
    Wing.chord_mid = chord_break
    Wing.areas.gross = 2.0*planform_wing.area
    Wing.areas.exposed = 2.0*planform_exposed.area
    Wing.areas.affected = 2.0*planform_flapped.area
    Wing.areas.wetted = 2.0*(1+0.2*thickness_to_chord)*Wing.areas.exposed
    Wing.mac = planform_wing.mean_aerodynamic_chord
    Wing.aerodynamic_center = Wing.origin + np.array([x_ac_local, 0, 0])

    print(Wing)

    return 0
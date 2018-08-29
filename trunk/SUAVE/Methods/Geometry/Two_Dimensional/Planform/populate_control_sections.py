# populate_control_sections.
#
# Created:  Jan 2015, T. Momose
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Components.Wings.Control_Surface import Control_Surface

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
def populate_control_sections(control_surface,span_fractions,chord_fractions,relative_twists,wing):
    """
    Creates Control_Surface_Sections defining a control surface such that:
        -There are as many Control_Surface_Sections as the length of
        span_fractions
        -The i-th Control_Surface_Section has the local chord fraction
        given in chord_fractions[i], the twist given in twists[i], and is
        at the spanwise position, span_fractions[i]
        -The control surface origin is defined based on the geometry of the
        wing (i.e., dimensional projected span, LE sweep, root chord, taper)

    Preconditions:
        -span_fractions has as many sections as will be used to define the
        control surface (at least two)
        -relative_twists has the same length as span_fractions and
        indicates the twist in the local chord frame of reference - twist
        is zero if the control surface chord line is parallel to the local
        chord line in the undeflected state.
        -span_fractions and chord_fractions have the same length and
        indicate the y- and x-coordinates of the section leading edge as
        fractions of wingspan and local chord, repsectively

    Postconditions:
        -control_surface.sections contains len(span_fractions)
        Control_Surface_Sections with size and position parameters filled
        -Returns control_surface with the specified sections appended

    Assumes a trailing-edge control surface
    """

    if len(span_fractions) < 2:
        raise ValueError('Two or more sections required for control surface definition')
    
    assert len(span_fractions) == len(chord_fractions) == len(relative_twists) , 'dimension array length mismatch'

    sw   = wing.sweeps.quarter_chord
    di   = wing.dihedral
    span = wing.spans.projected
    c_r  = wing.chords.root
    tpr  = wing.taper
    orig = wing.origin

    inboard = Control_Surface_Section()
    inboard.tag = 'inboard_section'
    inboard.origins.span_fraction  = span_fractions[0]
    inboard.origins.chord_fraction = 1. - chord_fractions[0]
    local_chord = c_r * (1 + 2. * span_fractions[0] * (tpr - 1))
    inboard.origins.dimensional[0] = orig[0] + span*span_fractions[0]*np.tan(sw) + local_chord*inboard.origins.chord_fraction
    inboard.origins.dimensional[1] = orig[1] + span*span_fractions[0]
    inboard.origins.dimensional[2] = orig[2] + span*span_fractions[0]*np.tan(di)
    inboard.chord_fraction = chord_fractions[0]
    inboard.twist = relative_twists[0]
    control_surface.append_section(inboard)

    outboard = Control_Surface_Section()
    outboard.tag = 'outboard_section'
    outboard.origins.span_fraction  = span_fractions[-1]
    outboard.origins.chord_fraction = 1. - chord_fractions[-1]
    local_chord = c_r * (1 + 2. * span_fractions[-1] * (tpr - 1))
    outboard.origins.dimensional[0] = orig[0] + span*span_fractions[-1]*np.tan(sw) + local_chord*outboard.origins.chord_fraction
    outboard.origins.dimensional[1] = orig[1] + span*span_fractions[-1]
    outboard.origins.dimensional[2] = orig[2] + span*span_fractions[-1]*np.tan(di)
    outboard.chord_fraction = chord_fractions[-1]
    outboard.twist = relative_twists[-1]
    control_surface.append_section(outboard)

    control_surface.span_fraction = abs(span_fractions[-1] - span_fractions[0])

    if len(span_fractions) > 2:
        i = 1
        while i+1 < len(span_fractions):
            section = Control_Surface_Section()
            section.tag = ('inner_section{}'.format(i))
            section.origins.span_fraction  = span_fractions[i]
            section.origins.chord_fraction = 1. - chord_fractions[i]
            local_chord = c_r * (1 + 2. * span_fractions[i] * (tpr - 1))
            section.origins.dimensional[0] = orig[0] + span*span_fractions[i]*np.tan(sw) + local_chord*section.origins.chord_fraction
            section.origins.dimensional[1] = orig[1] + span*span_fractions[i]
            section.origins.dimensional[2] = orig[2] + span*span_fractions[i]*np.tan(di)
            section.chord_fraction = chord_fractions[i]
            section.twist = relative_twists[i]
            control_surface.append_section(section)
            i += 1

    return control_surface
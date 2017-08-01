## @ingroup Methods-Geometry-Three_Dimensional
# compute_chord_length_from_span_location.py
# 
# Created:  Oct 2015, M. Vegh, 
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Compute Chord Length from Span Location
# ---------------------------------------------------------------------- 

## @ingroup Methods-Geometry-Three_Dimensional
def compute_chord_length_from_span_location(wing,span_location):
    """Computes the chord length given a location along the half-span.

    Assumptions:
    Linear variation of chord with span.

    Source:
    None

    Inputs:
    wing.chords.
      root                [m]
      tip                 [m]
    wing.spans.projected  [m]
    span_location         [m]

    Outputs:
    chord_length          [m]

    Properties Used:
    N/A
    """      

    chord_span_slope=(.25*wing.chords.root-.25*wing.chords.tip)/(wing.spans.projected*.5)
    chord_length=4*span_location*chord_span_slope+wing.chords.tip
    
    return chord_length
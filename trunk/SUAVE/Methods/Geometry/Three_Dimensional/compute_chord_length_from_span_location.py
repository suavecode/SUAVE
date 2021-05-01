## @ingroup Methods-Geometry-Three_Dimensional
# compute_chord_length_from_span_location.py
# 
# Created:  Oct 2015, M. Vegh, 
# Modified: Jan 2016, E. Botero
# Modified: Jan 2019, E. Botero

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
    #unpack
    ct = wing.chords.tip
    cr = wing.chords.root
    b  = wing.spans.projected
    
    b_2 = b/2.
    
    chord_length = ct + ((cr-ct)/b_2)*(b_2-span_location)

    
    return chord_length
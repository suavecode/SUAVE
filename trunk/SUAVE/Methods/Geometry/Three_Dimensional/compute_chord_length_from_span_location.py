# compute_chord_length_from_span_location.py
# 
# Created:  Oct 2015, M. Vegh, 
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Compute Chord Length from Span Location
# ---------------------------------------------------------------------- 

def compute_chord_length_from_span_location(wing,span_location):
    '''
    Computes the chord length given a location along the half-span.
    Assumes linear variation of chord with span
    '''    

    chord_span_slope=(.25*wing.chords.root-.25*wing.chords.tip)/(wing.spans.projected*.5)
    chord_length=4*span_location*chord_span_slope+wing.chords.tip
    
    return chord_length
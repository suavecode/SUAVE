# compute_span_location_from_chord_length.py
# 
# Created:  Oct 2015, M. Vegh, 
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Compute Span Location from Chord Length
# ---------------------------------------------------------------------- 

def compute_span_location_from_chord_length(wing,chord_length):
    '''
    computes the location along the half-span from a specified chord length
    (e.g. mean aerodynamic chord). Assumes linear variation of chord with span
    '''    

    chord_span_slope=(.25*wing.chords.root-.25*wing.chords.tip)/(wing.spans.projected/2.)
    
    if chord_span_slope==0:  #prevent divide by zero errors (happens if taper=1)
        span_location=0
    else:
        span_location=(.25*chord_length-.25*wing.chords.tip)/chord_span_slope 
        
    return span_location
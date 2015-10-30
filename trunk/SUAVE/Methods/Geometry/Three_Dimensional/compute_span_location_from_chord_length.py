#compute_span_location_from_chord_length

#Created: M. Vegh, Oct. 2015
 
import numpy as np 
def compute_span_location_from_chord_length(wing,chord_length):
    chord_span_slope=(.25*wing.chords.root-.25*wing.chords.tip)/(wing.spans.projected/2.)
    span_location=(.25*chord_length-.25*wing.chords.tip)/chord_span_slope 
    return span_location
#compute_span_location_from_chord_length

#Created: M. Vegh, Oct. 2015
 
import numpy as np 
def compute_chord_length_from_span_location(wing,span_location):
    chord_span_slope=(.25*wing.chords.root-.25*wing.chords.tip)/(wing.spans.projected*.5)
    #span_location=(.25*chord_length-.25*wing.chords.tip)/chord_span_slope 
    chord_length=4*span_location*chord_span_slope+wing.chords.tip
    return chord_length
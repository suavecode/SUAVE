## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
# vortex_lift.py
# 
# Created:  Jub 2014, T. MacDonald
# Modified: Jul 2014, T. MacDonald
#           Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#   Vortex Lift
# ----------------------------------------------------------------------

## @ingroup Methods-Aerodynamics-Fidelity_Zero-Lift
def vortex_lift(AoA,configuration,wing):
    """Computes vortex lift

    Assumptions:
    wing capable of vortex lift

    Source:
    adg.stanford.edu (Stanford AA241 A/B Course Notes)
    
    Inputs:
    wing.aspect_ratio         [Unitless]
    wing.sweeps.quarter_chord [radians]

    Outputs:
    CL_prime  (added CL)      [Unitless]

    Properties Used:
    N/A
    """  
    #-------------------------------------------------------------------------
    # OLD CODE
    #
    AR    = wing.aspect_ratio
    GAMMA = wing.sweeps.quarter_chord
    
    # angle of attack
    a = AoA
    
    # lift coefficient addition
    CL_prime = np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(GAMMA)-np.sin(a)/(2*np.cos(GAMMA)))
    #-------------------------------------------------------------------------    

    #if len(wing.Segments.keys())>0: 
        #symm                 = wing.symmetric
        #semispan             = wing.spans.projected*0.5 * (2 - symm)
        #root_chord           = wing.chords.root
        #segment_percent_span = 0;   
        #num_segments           = len(wing.Segments.keys())      
    
        #for i_segs in xrange(num_segments):         
            #wing_segment_spans_projected      = semispan*(wing.Segments[i_segs+1].percent_span_location - wing.Segments[i_segs].percent_span_location )
            #wing_segment_chords_root          = root_chord*wing.Segments[i_segs].root_chord_percent
            #wing_segment_chords_tip           = root_chord*wing.Segments[i_segs+1].root_chord_percent
            #wing_segment_sweeps_quarter_chord = wing.Segments[i_segs].sweeps.quarter_chord
            #wing_segment_areas_reference      = wing_segment_spans_projected *(wing_segment_chords_root+wing_segment_chords_tip)*0.5
            #wing_segment_aspect_ratio         = (wing_segment.spans.projected)**2/wing_segment_areas_reference    

            #AR    = wing_segment_aspect_ratio 
            #GAMMA = wing_segment_sweeps_quarter_chord 
            
            ## angle of attack
            #a = AoA
            
            ## lift coefficient addition
            #CL_prime += np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(GAMMA)-np.sin(a)/(2*np.cos(GAMMA)))
   
   
   
    #else:            
        #AR    = wing.aspect_ratio
        #GAMMA = wing.sweeps.quarter_chord
        
        ## angle of attack
        #a = AoA
        
        ## lift coefficient addition
        #CL_prime = np.pi*AR/2*np.sin(a)*np.cos(a)*(np.cos(a)+np.sin(a)*np.cos(a)/np.cos(GAMMA)-np.sin(a)/(2*np.cos(GAMMA)))
        
    return CL_prime
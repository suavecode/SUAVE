
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from copy import deepcopy


# ----------------------------------------------------------------------
#  Expand Sub Segments
# ----------------------------------------------------------------------
          
def expand_sub_segments(segment,state):
    
    last_tag = None
    
    for tag,sub_segment in segment.sub_segments.items():
        
        sub_state = deepcopy( sub_segment.state )
        sub_segment.initialize(sub_state)
        
        state.sub_segments[tag] = sub_state
        state.unknowns[tag]     = sub_state.unknowns
        state.conditions[tag]   = sub_state.conditions
        state.residuals[tag]    = sub_state.residuals
        
        if last_tag:
            state.sub_segments[tag].initials = state.sub_segments[last_tag]
        last_tag = tag
        

# ----------------------------------------------------------------------
#  Update Sub Segments
# ----------------------------------------------------------------------        

def update_sub_segments(segment,state):
    for tag,sub_segment in segment.sub_segments.items():
        sub_segment.iterate(state.sub_segments[tag])
        
                    
# ----------------------------------------------------------------------
#  Finalize Sub Segments
# ----------------------------------------------------------------------

def finalize_sub_segments(segment,state):
    
    from SUAVE.Analyses.New_Segment.Conditions import Conditions
    
    for tag,sub_segment in segment.sub_segments.items():
        sub_segment.finalize(state.sub_segments[tag])
        state.sub_segments[tag].initials = Conditions()
    
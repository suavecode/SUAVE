
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from copy import deepcopy


# ----------------------------------------------------------------------
#  Expand Sub Segments
# ----------------------------------------------------------------------
          
def expand_sub_segments(segment,state):
    
    last_tag = None
    
    for tag,sub_segment in segment.segments.items():
        
        sub_state = deepcopy( sub_segment.state )
        sub_segment.initialize(sub_state)
        
        state.segments[tag]     = sub_state
        state.unknowns[tag]     = sub_state.unknowns
        state.conditions[tag]   = sub_state.conditions
        state.residuals[tag]    = sub_state.residuals
        
        if last_tag:
            state.segments[tag].initials = state.segments[last_tag]
        last_tag = tag
        

# ----------------------------------------------------------------------
#  Update Sub Segments
# ----------------------------------------------------------------------        

def update_sub_segments(segment,state):
    for tag,sub_segment in segment.segments.items():
        sub_segment.iterate(state.segments[tag])
        
                    
# ----------------------------------------------------------------------
#  Finalize Sub Segments
# ----------------------------------------------------------------------

def finalize_sub_segments(segment,state):
    
    from SUAVE.Analyses.Missions.Segments.Conditions import Conditions
    
    for tag,sub_segment in segment.segments.items():
        sub_segment.finalize(state.segments[tag])
        state.segments[tag].initials = Conditions()
    
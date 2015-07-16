

# ----------------------------------------------------------------------
#  Expand State
# ----------------------------------------------------------------------

def expand_state(segment,state):

    n_points = state.numerics.number_control_points
    
    state.expand_rows(n_points)
    
    return
    
## @ingroup methods-mission-segments
# expand_state.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Expand State
# ----------------------------------------------------------------------

## @ingroup methods-mission-segments
def expand_state(segment,state):
    
    """Makes all vectors in the state the same size.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.numerics.number_control_points  [Unitless]

    Outputs:
    N/A

    Properties Used:
    N/A
    """       

    n_points = state.numerics.number_control_points
    
    state.expand_rows(n_points)
    
    return
    
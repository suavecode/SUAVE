""" evaluate_mission.py: solve Mission segments in sequence """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import copy
from evaluate_segment import evaluate_segment

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def evaluate_mission(mission):

    mission = copy.deepcopy(mission)
    segments = mission.Segments
    
    # evaluate each segment 
    for i,segment in enumerate(segments.values()):
        
        if i > 0:
            # link segment final conditions with initial conditions
            segment.initials = segments[i-1].get_final_conditions()
        else:
            segment.initials = None
            
        # run segment
        evaluate_segment(segment)
        
    return mission


# ----------------------------------------------------------------------
#  Helper Methods
# ----------------------------------------------------------------------

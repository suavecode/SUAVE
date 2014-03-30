""" evaluate_mission.py: solve Mission segments in sequence """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import copy
from SUAVE.Methods.Solvers import pseudospectral

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def evaluate_mission(mission):

    results = copy.deepcopy(mission)

    # evaluate each segment 
    for i in range(len(results.Segments)):

        segment = results.Segments[i]
        # print mission.Segments[i]

        # determine ICs for this segment
        if i == 0:                                              # first segment of mission
            segment.m0 = results.m0
            segment.t0 = 0.0
        else:                                                   # inherit ICs from previous segment
            segment.m0 = results.Segments[i-1].m[-1]
            segment.t0 = results.Segments[i-1].t[-1]

        # run segment
        pseudospectral(segment)
        #add horizontal distance covered
        if i !=0:
            segment.vectors.r[:,0]+=results.Segments[i-1].vectors.r[-1,0]

    return results


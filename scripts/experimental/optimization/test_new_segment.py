

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import full_setup

import SUAVE
from SUAVE.Core import Units

from copy import deepcopy

from time import time


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    
    vehicle  = full_setup.vehicle_setup()
    configs  = full_setup.configs_setup(vehicle)
    
    analyses = full_setup.analyses_setup(configs)
    
    
    vehicle  = configs.base
    configs.finalize()
    
    analyses = analyses.base
    analyses.finalize()
    
    segment = SUAVE.Analyses.New_Segment.Cruise.Cruise()
    segment.analyses.extend(analyses)
    
    
    segment.altitude  = 10.668  * Units.km
    segment.air_speed = 230.412 * Units['m/s']
    segment.distance  = 3933.65 * Units.km
    
    state1 = deepcopy( segment.state )    
    state2 = deepcopy( segment.state )    
    
    tic = time()
    # once!
    segment.evaluate( state1 )
    print state1.conditions.weights.total_mass[-1,0]
    
    # again!
    segment.evaluate( state2 )
    print state2.conditions.weights.total_mass[-1,0]
    
    print 't' , time()-tic
    
    
    # segment container!
    
    segment_1 = deepcopy(segment)
    segment_2 = deepcopy(segment)
    
    mission = SUAVE.Analyses.New_Segment.Cruise.Cruise.Container()
    
    mission.sub_segments.segment_1 = segment_1
    mission.sub_segments.segment_2 = segment_2
    
    state = deepcopy( mission.state )
    
    tic = time()
    
    mission.evaluate( state )    
    
    print 't', time()-tic
    
    return



if __name__ == '__main__':
    main()

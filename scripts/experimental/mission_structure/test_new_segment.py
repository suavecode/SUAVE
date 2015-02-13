

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
    
    segment = SUAVE.Analyses.Missions.Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.analyses.extend(analyses)
    
    
    segment.altitude  = 10.668  * Units.km
    segment.air_speed = 230.412 * Units['m/s']
    segment.distance  = 3933.65 * Units.km
    
    tic = time()
    # once!
    state1 = segment.evaluate()
    print state1.conditions.weights.total_mass[-1,0]
    
    # again!
    state2 = segment.evaluate()
    print state2.conditions.weights.total_mass[-1,0]
    
    print 't' , time()-tic
    
    # Ok now do a climb segment
    csegment = SUAVE.Analyses.Missions.Segments.Climb.Constant_Speed_Constant_Rate()
    csegment.altitude_start = 0.0
    csegment.altitude_end   = 10* Units.km
    csegment.climb_rate     = 3.  * Units.m / Units.s
    csegment.air_speed      = 230.412 * Units['m/s']
    csegment.analyses.extend(analyses)
    
    state3 = csegment.evaluate()
    print state3.conditions.weights.total_mass
    print state3.conditions.freestream.altitude
    
    ## segment container!
    
    segment_1 = deepcopy(segment)
    segment_2 = deepcopy(segment)
    
    mission = SUAVE.Analyses.Missions.Mission()
    
    mission.segments.segment_1 = segment_1
    mission.segments.segment_2 = segment_2
    
    tic = time()
    
    state4 = mission.evaluate( )    
    
    print 't', time()-tic
    print state4.merged().conditions.weights.total_mass
    
    return



if __name__ == '__main__':
    main()

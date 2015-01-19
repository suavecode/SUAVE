

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import full_setup

import SUAVE
from SUAVE.Core import Units

from copy import deepcopy


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
    
    state = deepcopy( segment.state )
    
    segment.evaluate( state )
    
    return



if __name__ == '__main__':
    main()

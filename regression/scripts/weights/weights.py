# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup
from SUAVE.Methods.Performance  import payload_range


def main():
  
    vehicle = vehicle_setup()    
    weight = Tube_Wing.empty(vehicle)
    
    # regression values    
    actual = Data()
    actual.payload         = 27349.9081525 #includes cargo #17349.9081525 #without cargo
    actual.pax             = 15036.587065500002
    actual.bag             = 2313.3210870000003
    actual.fuel            = 12990.957450008464 #includes cargo #22177.6377131 #without cargo
    actual.empty           = 38674.934397491539
    actual.wing            = 6649.7096587384294
    actual.fuselage        = 6642.0611642718986
    actual.propulsion      = 6838.1851749566231
    actual.landing_gear    = 3160.632
    actual.systems         = 13479.10479056802
    actual.wt_furnish      = 6431.80372889
    actual.horizontal_tail = 1024.5873332662029
    actual.vertical_tail   = 629.03876835025949
    actual.rudder          = 251.61550734010382
    
    # error calculations
    error                 = Data()
    error.payload         = (actual.payload - weight.payload)/actual.payload
    error.pax             = (actual.pax - weight.pax)/actual.pax
    error.bag             = (actual.bag - weight.bag)/actual.bag
    error.fuel            = (actual.fuel - weight.fuel)/actual.fuel
    error.empty           = (actual.empty - weight.empty)/actual.empty
    error.wing            = (actual.wing - weight.wing)/actual.wing
    error.fuselage        = (actual.fuselage - weight.fuselage)/actual.fuselage
    error.propulsion      = (actual.propulsion - weight.propulsion)/actual.propulsion
    error.landing_gear    = (actual.landing_gear - weight.landing_gear)/actual.landing_gear
    error.systems         = (actual.systems - weight.systems)/actual.systems
    error.wt_furnish      = (actual.wt_furnish - weight.systems_breakdown.furnish)/actual.wt_furnish
    error.horizontal_tail = (actual.horizontal_tail - weight.horizontal_tail)/actual.horizontal_tail
    error.vertical_tail   = (actual.vertical_tail - weight.vertical_tail)/actual.vertical_tail
    error.rudder          = (actual.rudder - weight.rudder)/actual.rudder
    
    print 'Results (kg)'
    print weight
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1e-6)    
   
    
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()    
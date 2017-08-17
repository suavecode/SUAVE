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
    
    
    actual = Data()
    actual.payload         = 27349.9081525 #includes cargo #17349.9081525 #without cargo
    actual.pax             = 15036.5870655
    actual.bag             = 2313.321087
    actual.fuel            = 12973.7402681 #includes cargo #22177.6377131 #without cargo
    actual.empty           = 38692.1515794
    actual.wing            = 6649.70965874
    actual.fuselage        = 6641.4815082
    actual.propulsion      = 6855.68572746
    actual.landing_gear    = 3160.632
    actual.systems         = 13479.1047906
    actual.wt_furnish      = 6431.80372889
    actual.horizontal_tail = 1024.58733327
    actual.vertical_tail   = 629.03876835
    actual.rudder          =  251.61550734
    
    
    '''
    #old errors; original geometry appears to be incorrect
    actual.payload = 17349.9081525
    actual.pax = 15036.5870655
    actual.bag = 2313.321087
    actual.fuel = -13680.6265874
    actual.empty = 75346.5184349
    actual.wing = 27694.192985
    actual.fuselage = 11423.9380852
    actual.propulsion = 6855.68572746 
    actual.landing_gear = 3160.632
    actual.systems = 16655.7076511
    actual.wt_furnish = 7466.1304102
    actual.horizontal_tail = 2191.30720639
    actual.vertical_tail = 5260.75341411
    actual.rudder = 2104.30136565    
    '''
    
    
    error = Data()
    error.payload = (actual.payload - weight.payload)/actual.payload
    error.pax = (actual.pax - weight.pax)/actual.pax
    error.bag = (actual.bag - weight.bag)/actual.bag
    error.fuel = (actual.fuel - weight.fuel)/actual.fuel
    error.empty = (actual.empty - weight.empty)/actual.empty
    error.wing = (actual.wing - weight.wing)/actual.wing
    error.fuselage = (actual.fuselage - weight.fuselage)/actual.fuselage
    error.propulsion = (actual.propulsion - weight.propulsion)/actual.propulsion
    error.landing_gear = (actual.landing_gear - weight.landing_gear)/actual.landing_gear
    error.systems = (actual.systems - weight.systems)/actual.systems
    error.wt_furnish = (actual.wt_furnish - weight.systems_breakdown.furnish)/actual.wt_furnish
    
    error.horizontal_tail = (actual.horizontal_tail - weight.horizontal_tail)/actual.horizontal_tail
    error.vertical_tail = (actual.vertical_tail - weight.vertical_tail)/actual.vertical_tail
    error.rudder = (actual.rudder - weight.rudder)/actual.rudder
    
    print 'Results (kg)'
    print weight
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<0.001)    
   
    
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()    
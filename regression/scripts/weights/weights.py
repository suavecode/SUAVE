# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Methods.Weights.Correlations import General_Aviation as General_Aviation
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup
from Cessna_172 import vehicle_setup as vehicle_setup_general_aviation


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
    
    print 'Results tube and wing (kg)'
    print weight
    
    print 'Relative Errors'
    print error  
    
    
      
    for k,v in error.items():
        assert(np.abs(v)<0.001)    
   
    #General Aviation weights; note that values are taken from Raymer,
    #but there is a huge spread among the GA designs, so individual components
    #differ a good deal from the actual design
   
    vehicle        = vehicle_setup_general_aviation()
    GTOW           = vehicle.mass_properties.max_takeoff
    weight         = General_Aviation.empty(vehicle)
    weight.fuel    = vehicle.fuel.mass_properties.mass 
    actual         = Data()
    actual.bag     = 0.
    actual.empty   = 618.485310343
    actual.fuel    = 144.69596603

    actual.wing            = 124.673093906
    actual.fuselage        = 119.522072873
    actual.propulsion      = 194.477769922 #includes power plant and propeller, does not include fuel system
    actual.landing_gear    = 44.8033840543+5.27975390045
    actual.furnishing      = 37.8341395817
    actual.electrical      = 36.7532226254
    actual.control_systems = 14.8331955546
    actual.fuel_systems    = 15.6859717453
    actual.systems         = 108.096549345

    error = Data()
    error.fuel = (actual.fuel - weight.fuel)/actual.fuel
    error.empty = (actual.empty - weight.empty)/actual.empty
    error.wing = (actual.wing - weight.wing)/actual.wing
    error.fuselage = (actual.fuselage - weight.fuselage)/actual.fuselage
    error.propulsion = (actual.propulsion - weight.propulsion)/actual.propulsion
    error.landing_gear = (actual.landing_gear - (weight.landing_gear_main+weight.landing_gear_nose))/actual.landing_gear
    error.furnishing = (actual.furnishing-weight.systems_breakdown.furnish)/actual.furnishing
    error.electrical = (actual.electrical-weight.systems_breakdown.electrical)/actual.electrical
    error.control_systems = (actual.control_systems-weight.systems_breakdown.control_systems)/actual.control_systems
    error.fuel_systems = (actual.fuel_systems-weight.systems_breakdown.fuel_system)/actual.fuel_systems
    error.systems = (actual.systems - weight.systems)/actual.systems

    print 'actual.systems=', actual.systems
    print 'General Aviation Results (kg)'
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
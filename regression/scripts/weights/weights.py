# weights.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Tube_Wing as Tube_Wing
from SUAVE.Methods.Weights.Correlations import General_Aviation as General_Aviation
from SUAVE.Methods.Weights.Correlations import BWB as BWB
from SUAVE.Methods.Weights.Correlations import Human_Powered as HP

from SUAVE.Core import (Data, Container,)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

import sys

sys.path.append('../Vehicles')
# the analysis functions

from Boeing_737 import vehicle_setup
from Cessna_172 import vehicle_setup as vehicle_setup_general_aviation
from BWB import vehicle_setup  as bwb_setup
from Solar_UAV import vehicle_setup  as hp_setup


def main():
  
    vehicle = vehicle_setup()    
    weight = Tube_Wing.empty(vehicle)
    
    # regression values    
    actual = Data()
    actual.payload         = 27349.9081525 #includes cargo #17349.9081525 #without cargo
    actual.pax             = 15036.587065500002
    actual.bag             = 2313.3210870000003
    actual.fuel            = 12977.803363592691 #includes cargo #22177.6377131 #without cargo
    actual.empty           = 38688.08848390731
    actual.wing            = 6649.709658738429
    actual.fuselage        = 6642.061164271899
    actual.propulsion      = 6838.185174956626
    actual.landing_gear    = 3160.632
    actual.systems         = 13479.10479056802
    actual.wt_furnish      = 6431.80372889
    actual.horizontal_tail = 1037.7414196819743
    actual.vertical_tail   = 629.0387683502595
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
    
    print('Results (kg)')
    print(weight)
    
    print('Relative Errors')
    print(error)  
      
    for k,v in list(error.items()):
        assert(np.abs(v)<1E-6)    
   
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

    error                 = Data()
    error.fuel            = (actual.fuel - weight.fuel)/actual.fuel
    error.empty           = (actual.empty - weight.empty)/actual.empty
    error.wing            = (actual.wing - weight.wing)/actual.wing
    error.fuselage        = (actual.fuselage - weight.fuselage)/actual.fuselage
    error.propulsion      = (actual.propulsion - weight.propulsion)/actual.propulsion
    error.landing_gear    = (actual.landing_gear - (weight.landing_gear_main+weight.landing_gear_nose))/actual.landing_gear
    error.furnishing      = (actual.furnishing-weight.systems_breakdown.furnish)/actual.furnishing
    error.electrical      = (actual.electrical-weight.systems_breakdown.electrical)/actual.electrical
    error.control_systems = (actual.control_systems-weight.systems_breakdown.control_systems)/actual.control_systems
    error.fuel_systems    = (actual.fuel_systems-weight.systems_breakdown.fuel_system)/actual.fuel_systems
    error.systems         = (actual.systems - weight.systems)/actual.systems

    print('actual.systems=', actual.systems)
    print('General Aviation Results (kg)')
    print(weight)

    print('Relative Errors')
    print(error)  

    for k,v in list(error.items()):
        assert(np.abs(v)<1e-6)    

    # BWB WEIGHTS
    vehicle = bwb_setup()    
    weight  = BWB.empty(vehicle)
            
    # regression values    
    actual = Data()
    actual.payload         = 27349.9081525 #includes cargo #17349.9081525 #without cargo
    actual.pax             = 15036.587065500002
    actual.bag             = 2313.3210870000003
    actual.fuel            = 24860.343951919327
    actual.empty           = 26805.547895580676
    actual.wing            = 6576.679767012152
    actual.fuselage        = 1.0
    actual.propulsion      = 1413.8593105126783
    actual.landing_gear    = 3160.632
    actual.systems         = 15654.376818055844
    actual.wt_furnish      = 8205.349895589
    
    # error calculations
    error                 = Data()
    error.payload         = (actual.payload - weight.payload)/actual.payload
    error.pax             = (actual.pax - weight.pax)/actual.pax
    error.bag             = (actual.bag - weight.bag)/actual.bag
    error.fuel            = (actual.fuel - weight.fuel)/actual.fuel
    error.empty           = (actual.empty - weight.empty)/actual.empty
    error.wing            = (actual.wing - weight.wing)/actual.wing
    error.fuselage        = (actual.fuselage - (weight.fuselage+1.0))/actual.fuselage
    error.propulsion      = (actual.propulsion - weight.propulsion)/actual.propulsion
    error.systems         = (actual.systems - weight.systems)/actual.systems
    error.wt_furnish      = (actual.wt_furnish - weight.systems_breakdown.furnish)/actual.wt_furnish
            
    print('Results (kg)')
    print(weight)
            
    print('Relative Errors')
    print(error)  
              
    for k,v in list(error.items()):
        assert(np.abs(v)<1E-6)    
    
    # Human Powered Aircraft
    vehicle = hp_setup()    
    weight = HP.empty(vehicle)
            
    # regression values    
    actual = Data()
    actual.empty           = 138.02737768459374
    actual.wing            = 89.86286881794777
    actual.fuselage        = 1.0
    actual.horizontal_tail = 31.749272074174737
    actual.vertical_tail   = 16.415236792471237
    
    # error calculations
    error                 = Data()
    error.empty           = (actual.empty - weight.empty)/actual.empty
    error.wing            = (actual.wing - weight.wing)/actual.wing
    error.fuselage        = (actual.fuselage - (weight.fuselage+1.0))/actual.fuselage
    error.horizontal_tail = (actual.horizontal_tail - weight.horizontal_tail)/actual.horizontal_tail
    error.vertical_tail   = (actual.vertical_tail - weight.vertical_tail)/actual.vertical_tail
            
    print('Results (kg)')
    print(weight)
    
    print('Relative Errors')
    print(error)  
              
    for k,v in list(error.items()):
        assert(np.abs(v)<1E-6)    



    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()    
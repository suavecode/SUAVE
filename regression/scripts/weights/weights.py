# weights.py
# Created:   
# Modified: Mar 2020, M. Clarke

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Weights.Correlations import Propulsion as Propulsion
from SUAVE.Methods.Weights.Correlations import Transport as Transport
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
    weight = Transport.empty(vehicle)
    
    # regression values    
    actual = Data()
    actual.payload         = 27349.9081525      # includes cargo #17349.9081525 #without cargo
    actual.pax             = 15036.587065500002
    actual.bag             = 2313.3210870000003
    actual.fuel            = 16504.32206450931  # includes cargo #22177.6377131 #without cargo
    actual.empty           = 35161.56978299069
    actual.wing            = 3461.869204335895
    actual.fuselage        = 6700.709511002648
    actual.propulsion      = 6838.185174956626
    actual.landing_gear    = 3160.632
    actual.systems         = 13390.723085494214
    actual.wt_furnish      = 6431.803728889001
    actual.horizontal_tail = 728.7965315109458
    actual.vertical_tail   = 880.6542756903633
    actual.rudder          = 251.61550734010382
    actual.nose_gear       = 316.06320000000005
    actual.main_gear       = 2844.5688
    
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
    error.nose_gear       = (actual.nose_gear - weight.nose_gear)/actual.nose_gear
    error.main_gear       = (actual.main_gear - weight.main_gear)/actual.main_gear
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
    actual.empty   = 720.1834370409678
    actual.fuel    = 144.69596603

    actual.wing            = 152.25407206578896
    actual.fuselage        = 126.7421108234472
    actual.propulsion      = 224.40728553408732
    actual.landing_gear    = 67.81320006645151
    actual.furnishing      = 37.8341395817
    actual.electrical      = 41.28649399649684
    actual.control_systems = 20.51671046011007
    actual.fuel_systems    = 20.173688786768366
    actual.systems         = 122.8010526627288

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
    actual.fuel            = 26119.117465169547
    actual.empty           = 25546.774382330455
    actual.wing            = 5317.906253761935
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
    actual.empty           = 143.59737768459374
    actual.wing            = 95.43286881794776
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
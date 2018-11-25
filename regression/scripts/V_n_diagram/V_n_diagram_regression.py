# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Performance  import V_n_diagram
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

import sys

sys.path.append('../Vehicles')
# the analysis functions

# Part 23
# aerobatic category
from Yak54 import vehicle_setup as vehicle_setup_Yak54
from  SU29 import vehicle_setup as vehicle_setup_SU29

# normal category
from  Cirrus_SR22 import vehicle_setup as vehicle_setup_SR22
from  Piper_M350  import vehicle_setup as vehicle_setup_Piper_M350

# commuter category
from   Tecnam_P2012  import vehicle_setup as vehicle_setup_Tecnam_P2012
from  DHC6_TwinOtter import vehicle_setup as vehicle_setup_DHC6_TwinOtter

# utility category
from   HAIG_Y12   import vehicle_setup as vehicle_setup_HAIG_Y12
from Pilatus_PC12 import vehicle_setup as vehicle_setup_Pilatus_PC12

# Part 25
from Boeing_737   import vehicle_setup as vehicle_setup_Boeing_B737

def main():

    analyses = SUAVE.Analyses.Vehicle()
    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)      
    
    altitude = 0 * Units.m
    delta_ISA = 0

    # Part 23 aircraft
    #--------------------------------------------------------------------------
    #------------------------------------------------
    # Aerobatic category aircraft regression test
    #------------------------------------------------

    # Yakovlev Yak - 54
    #---------------------------------------------
    vehicle  = vehicle_setup_Yak54()
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)

    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 57.923
    actual.Vs1_neg                  = 57.923
    actual.Va_pos                   = 173.769
    actual.Va_neg                   = 153.250
    actual.Vc                       = 199.834
    actual.Vd                       = 229.8
    actual.limit_load_pos           = 9
    actual.limit_load_neg           = -7
    actual.dive_limit_load_pos      = 9
    actual.dive_limit_load_neg      = -1.3149 
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg
        
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)    
    #--------------------------------------------------------------------------------------------------------------------
        
    # Sukhoi Su - 29
    #---------------------------------------------
    vehicle  = vehicle_setup_SU29()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 58.15307
    actual.Vs1_neg                  = 58.15307
    actual.Va_pos                   = 142.4454
    actual.Va_neg                   = 100.7241
    actual.Vc                       = 163.8122
    actual.Vd                       = 211.9856
    actual.limit_load_pos           = 6
    actual.limit_load_neg           = -3
    actual.dive_limit_load_pos      = 6
    actual.dive_limit_load_neg      = -1.12414
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg   
    
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2) 


    #------------------------------------------------
    # Normal category aircraft regression test
    #------------------------------------------------

    # Cirrus SR - 22
    #---------------------------------------------
    vehicle  = vehicle_setup_SR22()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 45.469
    actual.Vs1_neg                  = 54.752
    actual.Va_pos                   = 88.636
    actual.Va_neg                   = 67.503
    actual.Vc                       = 105.837
    actual.Vd                       = 147.217
    actual.limit_load_pos           = 4.535
    actual.limit_load_neg           = -2.535
    actual.dive_limit_load_pos      = 3.8
    actual.dive_limit_load_neg      = -1.458
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg
        
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)    
    #--------------------------------------------------------------------------------------------------------------------

    # Piper M350
    #---------------------------------------------
    vehicle  = vehicle_setup_Piper_M350()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 72.333
    actual.Vs1_neg                  = 76.8585
    actual.Va_pos                   = 140.5138
    actual.Va_neg                   = 94.42832
    actual.Vc                       = 165.3697
    actual.Vd                       = 227.7445
    actual.limit_load_pos           = 3.774
    actual.limit_load_neg           = -1.5095
    actual.dive_limit_load_pos      = 3.8
    actual.dive_limit_load_neg      = -0.46376
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg
        
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)    
    #--------------------------------------------------------------------------------------------------------------------

    #------------------------------------------------
    # Commuter category aircraft regression test
    #------------------------------------------------
    
    # Tecnam P2012
    #---------------------------------------------
    vehicle  = vehicle_setup_Tecnam_P2012()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 84.0859
    actual.Vs1_neg                  = 89.3462
    actual.Va_pos                   = 155.9128
    actual.Va_neg                   = 109.4263
    actual.Vc                       = 198.4436
    actual.Vd                       = 259.8548
    actual.limit_load_pos           = 3.438
    actual.limit_load_neg           = -1.5
    actual.dive_limit_load_pos      = 3.438
    actual.dive_limit_load_neg      = -0.3728
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg
       
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)    
    #--------------------------------------------------------------------------------------------------------------------


    # DHC-6 Twin Otter
    #---------------------------------------------
    vehicle  = vehicle_setup_DHC6_TwinOtter()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 79.2398
    actual.Vs1_neg                  = 84.1970
    actual.Va_pos                   = 141.008
    actual.Va_neg                   = 94.7606
    actual.Vc                       = 198.444
    actual.Vd                       = 246.861
    actual.limit_load_pos           = 3.5076
    actual.limit_load_neg           = -1.5076
    actual.dive_limit_load_pos      = 3.1667
    actual.dive_limit_load_neg      = -0.5597
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg
       
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)   
    #--------------------------------------------------------------------------------------------------------------------


    #------------------------------------------------
    # Utility category aircraft regression test
    #------------------------------------------------

    # HAIG - Y12
    #---------------------------------------------
    vehicle  = vehicle_setup_HAIG_Y12()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 81.4823
    actual.Vs1_neg                  = 86.5798
    actual.Va_pos                   = 170.919
    actual.Va_neg                   = 114.861
    actual.Vc                       = 198.444
    actual.Vd                       = 268.470
    actual.limit_load_pos           = 4.4
    actual.limit_load_neg           = -1.76
    actual.dive_limit_load_pos      = 4.4
    actual.dive_limit_load_neg      = -1
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg   
    
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)    
    #--------------------------------------------------------------------------------------------------------------------

    # Pilatus PC - 12
    #---------------------------------------------
    vehicle  = vehicle_setup_Pilatus_PC12()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 89.29371
    actual.Vs1_neg                  = 94.87986
    actual.Va_pos                   = 187.3041
    actual.Va_neg                   = 125.8724
    actual.Vc                       = 231.7575
    actual.Vd                       = 288.712
    actual.limit_load_pos           = 4.4
    actual.limit_load_neg           = -1.76
    actual.dive_limit_load_pos      = 4.4
    actual.dive_limit_load_neg      = -1
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg   
    
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)    
    #--------------------------------------------------------------------------------------------------------------------

    # Part 25 aircraft
    #--------------------------------------------------------------------------

    # Boeing B737
    #---------------------------------------------
    vehicle  = vehicle_setup_Boeing_B737()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 159.0879
    actual.Vs1_neg                  = 195.6588
    actual.Va_pos                   = 251.5401
    actual.Va_neg                   = 239.6321
    actual.Vc                       = 519.2608
    actual.Vd                       = 649.076
    actual.limit_load_pos           = 3.2668
    actual.limit_load_neg           = -1.5
    actual.dive_limit_load_pos      = 2.5
    actual.dive_limit_load_neg      = -0.41675
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1_pos)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1_neg)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va_pos)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va_neg)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_load_pos)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_load_neg)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.dive_limit_load_pos)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.dive_limit_load_neg)/actual.dive_limit_load_neg   
    
    print 'Results (kg)'
    print V_n_data
    
    print 'Relative Errors'
    print error  
      
    for k,v in error.items():
        assert(np.abs(v)<1E-2)   
    return

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()    

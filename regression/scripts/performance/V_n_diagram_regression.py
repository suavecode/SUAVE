# test_Weights.py

import SUAVE
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Performance import V_n_diagram
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing

import sys

sys.path.append('../Vehicles')
# the analysis functions

# Part 23
# aerobatic category
from Yak54_wing_only import vehicle_setup as vehicle_setup_Yak54
from  SU29_wing_only import vehicle_setup as vehicle_setup_SU29

# normal category
from  Cirrus_SR22_wing_only import vehicle_setup as vehicle_setup_SR22
from  Piper_M350_wing_only  import vehicle_setup as vehicle_setup_Piper_M350

# commuter category
from   Tecnam_P2012_wing_only  import vehicle_setup as vehicle_setup_Tecnam_P2012
from  DHC6_TwinOtter_wing_only import vehicle_setup as vehicle_setup_DHC6_TwinOtter

# utility category
from   HAIG_Y12_wing_only   import vehicle_setup as vehicle_setup_HAIG_Y12
from Pilatus_PC12_wing_only import vehicle_setup as vehicle_setup_Pilatus_PC12

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
    
    altitude  = 0 * Units.m
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
    actual.Vs1_pos                  = 57.93213
    actual.Vs1_neg                  = 57.93213
    actual.Va_pos                   = 173.79638
    actual.Va_neg                   = 153.274
    actual.Vc                       = 199.86584
    actual.Vd                       = 229.84571
    actual.limit_load_pos           = 9
    actual.limit_load_neg           = -7
    actual.dive_limit_load_pos      = 9
    actual.dive_limit_load_neg      = -1.313253
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg
    
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6)    
    #--------------------------------------------------------------------------------------------------------------------
        
    # Sukhoi Su - 29
    #---------------------------------------------
    vehicle  = vehicle_setup_SU29()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 58.1772
    actual.Vs1_neg                  = 58.1772
    actual.Va_pos                   = 142.5044
    actual.Va_neg                   = 100.7658
    actual.Vc                       = 163.8801
    actual.Vd                       = 212.0243
    actual.limit_load_pos           = 6
    actual.limit_load_neg           = -3
    actual.dive_limit_load_pos      = 6
    actual.dive_limit_load_neg      = -1.126836
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg 
    
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6) 


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
    actual.Vs1_pos                  = 45.47889
    actual.Vs1_neg                  = 54.76384
    actual.Va_pos                   = 88.65468
    actual.Va_neg                   = 67.51739
    actual.Vc                       = 105.8268
    actual.Vd                       = 147.2140
    actual.limit_load_pos           = 4.53441
    actual.limit_load_neg           = -2.534413
    actual.dive_limit_load_pos      = 3.8
    actual.dive_limit_load_neg      = -1.458333
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg
        
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6)    
    #--------------------------------------------------------------------------------------------------------------------

    # Piper M350
    #---------------------------------------------
    vehicle  = vehicle_setup_Piper_M350()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 72.3381
    actual.Vs1_neg                  = 76.8636
    actual.Va_pos                   = 140.52111
    actual.Va_neg                   = 94.433205
    actual.Vc                       = 165.35441
    actual.Vd                       = 227.75333
    actual.limit_load_pos           = 3.7735348
    actual.limit_load_neg           = -1.509414
    actual.dive_limit_load_pos      = 3.7735348
    actual.dive_limit_load_neg      = -0.4633023
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg
        
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6)    
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
    actual.Vs1_pos                  = 84.3000
    actual.Vs1_neg                  = 89.21476
    actual.Va_pos                   = 156.2416
    actual.Va_neg                   = 109.2653
    actual.Vc                       = 198.4253
    actual.Vd                       = 260.4157
    actual.limit_load_pos           = 3.435089
    actual.limit_load_neg           = -1.5
    actual.dive_limit_load_pos      = 3.435089
    actual.dive_limit_load_neg      = -0.3706622
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg
       
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6)    
    #--------------------------------------------------------------------------------------------------------------------


    # DHC-6 Twin Otter
    #---------------------------------------------
    vehicle  = vehicle_setup_DHC6_TwinOtter()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 79.2618
    actual.Vs1_neg                  = 84.2203
    actual.Va_pos                   = 141.0472
    actual.Va_neg                   = 94.7867
    actual.Vc                       = 198.4253
    actual.Vd                       = 246.9155
    actual.limit_load_pos           = 3.5066162
    actual.limit_load_neg           = -1.506615
    actual.dive_limit_load_pos      = 3.166656
    actual.dive_limit_load_neg      = -0.559585
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg
        
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6)   
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
    actual.Vs1_pos                  = 81.4788
    actual.Vs1_neg                  = 86.5761
    actual.Va_pos                   = 170.9114
    actual.Va_neg                   = 114.8561
    actual.Vc                       = 198.4253
    actual.Vd                       = 268.4589
    actual.limit_load_pos           = 4.4
    actual.limit_load_neg           = -1.76
    actual.dive_limit_load_pos      = 4.4
    actual.dive_limit_load_neg      = -1
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg
    
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6)    
    #--------------------------------------------------------------------------------------------------------------------

    # Pilatus PC - 12
    #---------------------------------------------
    vehicle  = vehicle_setup_Pilatus_PC12()    
    weight   = vehicle.mass_properties.max_takeoff
    V_n_data = V_n_diagram(vehicle,analyses,weight,altitude,delta_ISA)
    
    # regression values    
    actual                          = Data()
    actual.Vs1_pos                  = 89.2905
    actual.Vs1_neg                  = 94.8765
    actual.Va_pos                   = 187.2974
    actual.Va_neg                   = 125.8679
    actual.Vc                       = 231.4961
    actual.Vd                       = 288.7028
    actual.limit_load_pos           = 4.4
    actual.limit_load_neg           = -1.76
    actual.dive_limit_load_pos      = 4.4
    actual.dive_limit_load_neg      = -1
    
    # error calculations
    error                         = Data()
    error.Vs1_pos                 = (actual.Vs1_pos - V_n_data.Vs1.positive)/actual.Vs1_pos
    error.Vs1_neg                 = (actual.Vs1_neg - V_n_data.Vs1.negative)/actual.Vs1_neg
    error.Va_pos                  = (actual.Va_pos - V_n_data.Va.positive)/actual.Va_pos
    error.Va_neg                  = (actual.Va_neg - V_n_data.Va.negative)/actual.Va_neg
    error.Vc                      = (actual.Vc - V_n_data.Vc)/actual.Vc
    error.Vd                      = (actual.Vd - V_n_data.Vd)/actual.Vd
    error.limit_load_pos          = (actual.limit_load_pos - V_n_data.limit_loads.positive)/actual.limit_load_pos
    error.limit_load_neg          = (actual.limit_load_neg - V_n_data.limit_loads.negative)/actual.limit_load_neg
    error.dive_limit_load_pos     = (actual.dive_limit_load_pos - V_n_data.limit_loads.dive.positive)/actual.dive_limit_load_pos
    error.dive_limit_load_neg     = (actual.dive_limit_load_neg - V_n_data.limit_loads.dive.negative)/actual.dive_limit_load_neg
      
      
    for k,v in error.items():
        assert(np.abs(v)<1E-6)  


# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()    

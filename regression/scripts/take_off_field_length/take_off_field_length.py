# test_take_off_field_length.py
#
# Created:  Jun 2014, Tarik, Carlos, Celso
# Modified: Feb 2017, M. Vegh
#           Jan 2018, W. Maier
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Core import Units

from SUAVE.Methods.Performance.estimate_take_off_field_length import estimate_take_off_field_length


import sys
sys.path.append('../Vehicles')
from Embraer_190 import vehicle_setup, configs_setup

# package imports
import numpy as np
import pylab as plt

# ----------------------------------------------------------------------
#   Build the Vehicle
# ----------------------------------------------------------------------
def main():

    # ----------------------------------------------------------------------
    #   Main
    # ----------------------------------------------------------------------    
    vehicle = vehicle_setup()
    configs=configs_setup(vehicle)
    # --- Takeoff Configuration ---
    configuration = configs.takeoff
    configuration.wings['main_wing'].flaps_angle =  20. * Units.deg
    configuration.wings['main_wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    configuration.V2_VS_ratio = 1.21
    configuration.max_lift_coefficient_factor = 0.90
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.base = base_analysis(vehicle)

    # CLmax for a given configuration may be informed by user
    # configuration.maximum_lift_coefficient = 2.XX

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    w_vec = np.linspace(40000.,52000.,10)
    engines = (2,3,4)
    takeoff_field_length = np.zeros((len(w_vec),len(engines)))
    second_seg_clb_grad  = np.zeros((len(w_vec),len(engines)))
    
    compute_clb_grad = 1 # flag for Second segment climb estimation
    
    for id_eng,engine_number in enumerate(engines):
        
        configuration.propulsors.turbofan.number_of_engines = engine_number
        
        for id_w,weight in enumerate(w_vec):
            configuration.mass_properties.takeoff = weight
            takeoff_field_length[id_w,id_eng],second_seg_clb_grad[id_w,id_eng] = \
                    estimate_take_off_field_length(configuration,analyses,airport,compute_clb_grad)
    
    truth_TOFL = np.array([[1429.97653076,  911.43233599,  657.55600788],
                           [1519.77898248,  963.75506504,  694.61828231],
                           [1613.83669062, 1018.28810496,  733.16830719],
                           [1712.27298862, 1075.07387447,  773.22760301],
                           [1815.21566067, 1134.15607915,  814.81822348],
                           [1922.79702123, 1195.5797371 ,  857.96276768],
                           [2035.15399441, 1259.39120435,  902.68439205],
                           [2152.42819325, 1325.63820018,  949.00682226],
                           [2274.76599919, 1394.36983225,  996.95436495],
                           [2402.31864145, 1465.63662177, 1046.55191933]])                         
    print(' takeoff_field_length=',  takeoff_field_length)
    print(' second_seg_clb_grad = ', second_seg_clb_grad)                      
                             
    truth_clb_grad =  np.array([[0.09199351, 0.26583035, 0.4396672 ],
                                [0.08593675, 0.25371683, 0.42149692],
                                [0.08026935, 0.24238204, 0.40449473],
                                [0.07495541, 0.23175416, 0.38855291],
                                [0.06996329, 0.22176991, 0.37357653],
                                [0.06526498, 0.21237329, 0.3594816 ],
                                [0.06083563, 0.20351459, 0.34619355],
                                [0.05665309, 0.19514952, 0.33364594],
                                [0.05269758, 0.1872385 , 0.32177941],
                                [0.04895136, 0.17974605, 0.31054074]])

    TOFL_error = np.max(np.abs(truth_TOFL-takeoff_field_length)/truth_TOFL)                           
    GRAD_error = np.max(np.abs(truth_clb_grad-second_seg_clb_grad)/truth_clb_grad)
    
    print('Maximum Take OFF Field Length Error= %.4e' % TOFL_error)
    print('Second Segment Climb Gradient Error= %.4e' % GRAD_error)    
    
    import pylab as plt
    title = "TOFL vs W"
    plt.figure(1); 
    plt.plot(w_vec,takeoff_field_length[:,0], 'k-', label = '2 Engines')
    plt.plot(w_vec,takeoff_field_length[:,1], 'r-', label = '3 Engines')
    plt.plot(w_vec,takeoff_field_length[:,2], 'b-', label = '4 Engines')

    plt.title(title); plt.grid(True)
    plt.plot(w_vec,truth_TOFL[:,0], 'k--o', label = '2 Engines [truth]')
    plt.plot(w_vec,truth_TOFL[:,1], 'r--o', label = '3 Engines [truth]')
    plt.plot(w_vec,truth_TOFL[:,2], 'b--o', label = '4 Engines [truth]')
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Takeoff field length (m)')    
        
    
    title = "2nd Segment Climb Gradient vs W"
    plt.figure(2); 
    plt.plot(w_vec,second_seg_clb_grad[:,0], 'k-', label = '2 Engines')
    plt.plot(w_vec,second_seg_clb_grad[:,1], 'r-', label = '3 Engines')
    plt.plot(w_vec,second_seg_clb_grad[:,2], 'b-', label = '4 Engines')

    plt.title(title); plt.grid(True)
    plt.plot(w_vec,truth_clb_grad[:,0], 'k--o', label = '2 Engines [truth]')
    plt.plot(w_vec,truth_clb_grad[:,1], 'r--o', label = '3 Engines [truth]')
    plt.plot(w_vec,truth_clb_grad[:,2], 'b--o', label = '4 Engines [truth]')
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Second Segment Climb Gradient (%)')    
    
    assert( TOFL_error   < 1e-6 )
    assert( GRAD_error   < 1e-6 )

    return 
    
def base_analysis(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()
   
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy Analysis
    energy  = SUAVE.Analyses.Energy.Energy()
    energy.network=vehicle.propulsors
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)    
    
    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)     
    
    # done!
    return analyses    

# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()
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
    configs = configs_setup(vehicle)
    # --- Takeoff Configuration ---
    configuration = configs.takeoff
    configuration.wings['main_wing'].flaps_angle =  20. * Units.deg
    configuration.wings['main_wing'].slats_angle  = 25. * Units.deg
    # V2_V2_ratio may be informed by user. If not, use default value (1.2)
    configuration.V2_VS_ratio = 1.21
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses = base_analysis(vehicle)
    analyses.aerodynamics.settings.maximum_lift_coefficient_factor = 0.90

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
        
        configuration.networks.turbofan.number_of_engines = engine_number
        
        for id_w,weight in enumerate(w_vec):
            configuration.mass_properties.takeoff = weight
            takeoff_field_length[id_w,id_eng],second_seg_clb_grad[id_w,id_eng] = \
                    estimate_take_off_field_length(configuration,analyses,airport,compute_clb_grad)
    
    truth_TOFL =  np.array([[1150.16903333,  746.71052106,  540.38635856],
                            [1216.96378677,  786.27356816,  568.5974106 ],
                            [1286.74186571,  827.4394654 ,  597.90462262],
                            [1359.57593937,  870.23357085,  628.32103852],
                            [1435.54113401,  914.68194461,  659.85998817],
                            [1514.71507245,  960.811362  ,  692.53509396],
                            [1597.17791338, 1008.64932656,  726.36027706],
                            [1683.0123904 , 1058.22408289,  761.34976373],
                            [1772.303851  , 1109.56462935,  797.51809135],
                            [1865.14029531, 1162.70073066,  834.88011448]])

    
    print(' takeoff_field_length = ',  takeoff_field_length)
    print(' second_seg_clb_grad  = ', second_seg_clb_grad)                      
                             
    truth_clb_grad =  np.array([[0.06940517, 0.2464756 , 0.42354602],
                                [0.06347328, 0.23440741, 0.40534153],
                                [0.05790772, 0.22309881, 0.3882899 ],
                                [0.05267576, 0.21248089, 0.37228602],
                                [0.04774846, 0.20249278, 0.3572371 ],
                                [0.04310012, 0.19308054, 0.34306096],
                                [0.03870782, 0.18419618, 0.32968454],
                                [0.0345511 , 0.1757969 , 0.31704269],
                                [0.03061159, 0.16784435, 0.30507711],
                                [0.0268728 , 0.16030415, 0.29373549]])


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
    energy.network=vehicle.networks
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
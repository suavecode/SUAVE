# test_take_off_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified: M. Vegh, Feb 2017

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
    
   
    truth_TOFL = np.array([[ 1247.00542965 ,  804.01711004 ,  581.23537638],
                           [ 1321.66823558 ,  847.98266854 ,  612.51226535],
                           [ 1399.73808555 ,  893.75633559 ,  645.01865388],
                           [ 1481.30371728 ,  941.36890031 ,  678.77029442],
                           [ 1566.45694301 ,  990.85203446 ,  713.78330138],
                           [ 1655.29270092 , 1042.23830929 ,  750.0741594 ],
                           [ 1747.90910618 , 1095.56121225 ,  787.65973133],
                           [ 1844.40750193 , 1150.85516349 ,  826.55726616],
                           [ 1944.89251008 , 1208.15553232 ,  866.78440677],
                           [ 2049.47208207 , 1267.49865343 ,  908.35919766]])
                             
    truth_clb_grad =  np.array([[ 0.08489936 , 0.26124088 , 0.4375824 ],
                      [ 0.07877649 , 0.24899514 , 0.41921379],
                      [ 0.07304647 , 0.2375351  , 0.40202374],
                      [ 0.06767312 , 0.22678839 , 0.38590367],
                      [ 0.06262452 , 0.21669121 , 0.3707579 ],
                      [ 0.05787248 , 0.20718711 , 0.35650175],
                      [ 0.05339191 , 0.19822598 , 0.34306005],
                      [ 0.0491605  , 0.18976316 , 0.33036582],
                      [ 0.04515829 , 0.18175873 , 0.31835918],
                      [ 0.04136738 , 0.17417693 , 0.30698647]])
                               
                       
    TOFL_error = np.max(np.abs(truth_TOFL-takeoff_field_length))                           
    GRAD_error = np.max(np.abs(truth_clb_grad-second_seg_clb_grad))
    
    print 'Maximum Take OFF Field Length Error= %.4e' % TOFL_error
    print 'Second Segment Climb Gradient Error= %.4e' % GRAD_error    
    
    import pylab as plt
    title = "TOFL vs W"
    plt.figure(1); plt.hold
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
    plt.figure(2); plt.hold
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
    
    assert( TOFL_error   < 1e-5 )
    assert( GRAD_error   < 1e-5 )

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
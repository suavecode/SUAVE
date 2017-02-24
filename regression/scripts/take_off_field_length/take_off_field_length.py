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
    
   
    truth_TOFL = np.array([[ 1116.10635509 ,  726.47454766 ,  525.93957657],
                           [ 1180.15587618 ,  764.49121395 ,  553.07064632],
                           [ 1247.04359555 ,  804.03963238 ,  581.25141247],
                           [ 1316.83687778 ,  845.14335923 ,  610.49401596],
                           [ 1389.60534387 ,  887.82659431 ,  640.81085913],
                           [ 1465.42090714 ,  932.1141929  ,  672.21461168],
                           [ 1544.3578087  ,  978.0316776  ,  704.71821645],
                           [ 1626.49265282 , 1025.60525002 ,  738.33489512],
                           [ 1711.90444209 , 1074.86180226 ,  773.07815381],
                           [ 1800.67461244 , 1125.82892836 ,  808.96178847]]   )
 
                             
                             
    print ' takeoff_field_length=',  takeoff_field_length
    print ' second_seg_clb_grad = ', second_seg_clb_grad                      
                             
    truth_clb_grad =  np.array([[ 0.07441427 , 0.25242021 , 0.43042615],
                            [ 0.06825112 , 0.24009391 , 0.4119367 ],
                            [ 0.0624828  , 0.22855727 , 0.39463173],
                            [ 0.05707296 , 0.21773759 , 0.37840221],
                            [ 0.05198957 , 0.20757081 , 0.36315205],
                            [ 0.04720429 , 0.19800024 , 0.34879619],
                            [ 0.04269194 , 0.18897554 , 0.33525915],
                            [ 0.0384301  , 0.18045186 , 0.32247363],
                            [ 0.03439872 , 0.1723891  , 0.31037949],
                            [ 0.03057983 , 0.16475133 , 0.29892282]]  )

          
                       
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
# test_landing_field_length.py
#
# Created:  Tarik, Carlos, Celso, Jun 2014
# Modified: Emilio Botero 
#           Mar 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# SUave Imports
import SUAVE
from SUAVE.Core            import Data
from SUAVE.Core import Units
from SUAVE.Core import Units
from SUAVE.Methods.Performance.estimate_landing_field_length import estimate_landing_field_length
import sys
sys.path.append('../Vehicles')

from Embraer_190 import vehicle_setup, configs_setup



# package imports
import numpy as np
import pylab as plt

  
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
    
    # done!
    return analyses    



def main():

    # ----------------------------------------------------------------------
    #   Main
    # ----------------------------------------------------------------------

    # --- Vehicle definition ---
    vehicle = vehicle_setup()
    configs=configs_setup(vehicle)
    # --- Landing Configuration ---
    landing_config = configs.landing
    landing_config.wings['main_wing'].control_surfaces.flap.deflection = 30. * Units.deg
    landing_config.wings['main_wing'].control_surfaces.slat.deflection = 25. * Units.deg
    landing_config.wings['main_wing'].high_lift  = True
    # Vref_V2_ratio may be informed by user. If not, use default value (1.23)
    landing_config.Vref_VS_ratio = 1.23
    
    # CLmax for a given configuration may be informed by user
    # Used defined ajust factor for maximum lift coefficient
    analyses = base_analysis(vehicle)
    analyses.aerodynamics.settings.maximum_lift_coefficient_factor = 0.90

    # --- Airport definition ---
    airport = SUAVE.Attributes.Airports.Airport()
    airport.tag = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  SUAVE.Analyses.Atmospheric.US_Standard_1976()

    # =====================================
    # Landing field length evaluation
    # =====================================
    w_vec = np.linspace(20000.,44000.,10)
    landing_field_length = np.zeros_like(w_vec)
    for id_w,weight in enumerate(w_vec):
        landing_config.mass_properties.landing = weight
        landing_field_length[id_w] = estimate_landing_field_length(landing_config,analyses,airport)

    truth_LFL = np.array( [  723.67022689 ,  786.82625714 ,  849.98228739  , 913.13831764  , 976.29434789 , 1039.45037815 , 1102.6064084 ,  1165.76243865 , 1228.9184689 ,  1292.07449915])
    LFL_error = np.max(np.abs(landing_field_length-truth_LFL))
    
    print('Maximum Landing Field Length Error= %.4e' % LFL_error)
    
    title = "LFL vs W"
    plt.figure(1); 
    plt.plot(w_vec,landing_field_length, 'k-', label = 'Landing Field Length')

    plt.title(title); plt.grid(True)

    plt.figure(1); plt.plot(w_vec,truth_LFL, label = 'Landing Field Length (true)')
    legend = plt.legend(loc='lower right', shadow = 'true')
    plt.xlabel('Weight (kg)')
    plt.ylabel('Landing Field Length (m)')
    
    #assert( LFL_error   < 1e-5 )

    return 
    
# ----------------------------------------------------------------------        
#   Call Main
# ----------------------------------------------------------------------    

if __name__ == '__main__':
    main()
    plt.show()
        

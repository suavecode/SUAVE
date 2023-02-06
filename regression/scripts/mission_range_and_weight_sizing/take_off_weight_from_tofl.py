# take_off_weight_from_tofl.py
#
# Created:  Feb 2020 , M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# MARC Imports
import MARC
from MARC.Core            import Data
from MARC.Core import Units
from MARC.Core import Units
from MARC.Methods.Performance.find_take_off_weight_given_tofl import find_take_off_weight_given_tofl
import sys

sys.path.append('../Vehicles')

from Embraer_190 import vehicle_setup, configs_setup

# package imports
import numpy as np
import pylab as plt

  
# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():   
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)
    
    # --- Takeoff Configuration ---
    configuration                                = configs.takeoff
    configuration.wings['main_wing'].flaps_angle = 20. * Units.deg
    configuration.wings['main_wing'].slats_angle = 25. * Units.deg 
    configuration.V2_VS_ratio                    = 1.21
    analyses                                     = MARC.Analyses.Analysis.Container()
    analyses                                     = base_analysis(configuration)
    analyses.aerodynamics.settings.maximum_lift_coefficient_factor = 0.90
 
    # --- Airport definition ---
    airport            = MARC.Attributes.Airports.Airport()
    airport.tag        = 'airport'
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere =  MARC.Analyses.Atmospheric.US_Standard_1976() 
 
    # Set Tofl 
    target_tofl = 1487.92650289 
    
    # Compute take off weight given tofl
    max_tow = find_take_off_weight_given_tofl(configuration,analyses,airport,target_tofl)
    
    truth_max_tow = 46656.500266281124
    max_tow_error = np.max(np.abs(max_tow[0]-truth_max_tow)) 
    print('Range Error = %.4e' % max_tow_error)
    assert(max_tow_error   < 1e-6 )
    
    return  

    
def base_analysis(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = MARC.Analyses.Vehicle()
   
    #  Aerodynamics Analysis
    aerodynamics = MARC.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    analyses.append(aerodynamics)
    
    # ------------------------------------------------------------------
    #  Energy Analysis
    energy  = MARC.Analyses.Energy.Energy()
    energy.network=vehicle.networks
    analyses.append(energy)
    
    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = MARC.Analyses.Planets.Planet()
    analyses.append(planet)    
    
    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = MARC.Analyses.Atmospheric.US_Standard_1976()
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
        

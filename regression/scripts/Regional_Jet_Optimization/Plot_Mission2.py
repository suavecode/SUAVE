# Plot_Mission.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import MARC
from MARC.Core import Units
from MARC.Visualization import *     

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results):
    
    plot_altitude_sfc_weight(results) 
    
    plot_flight_conditions(results) 
    
    plot_aerodynamic_coefficients(results)  
    
    plot_aircraft_velocities(results)
    
    plot_stability_coefficients(results)
    
    plot_drag_components(results)


    return
 
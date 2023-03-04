
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from MARC.Visualization.Performance.Aerodynamics.Vehicle import *  
from MARC.Visualization.Performance.Mission              import *  
from MARC.Visualization.Performance.Aerodynamics.Rotor import *  
from MARC.Visualization.Performance.Energy.Fuel          import *   
from MARC.Visualization.Performance.Noise                import * 

def plot_mission(results): 
    
    plot_altitude_sfc_weight(results) 
    
    plot_flight_conditions(results) 
    
    plot_aerodynamic_coefficients(results)  
    
    plot_aircraft_velocities(results)
    
    plot_stability_coefficients(results)
    
    plot_drag_components(results)
 
    return
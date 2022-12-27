
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from SUAVE.Visualization.Performance.Aerodynamics.Vehicle import *  
from SUAVE.Visualization.Performance.Mission              import *  
from SUAVE.Visualization.Performance.Energy.Common        import *  
from SUAVE.Visualization.Performance.Energy.Fuel          import *   
from SUAVE.Visualization.Performance.Noise                import * 

def plot_mission(results): 
    
    plot_altitude_sfc_weight(results) 
    
    plot_flight_conditions(results) 
    
    plot_aerodynamic_coefficients(results)  
    
    plot_aircraft_velocities(results)
    
    plot_stability_coefficients(results)
    
    plot_drag_components(results)
 
    return
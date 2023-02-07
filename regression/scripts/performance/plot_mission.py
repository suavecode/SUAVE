
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

from MARC.Visualization.Performance.Aerodynamics.Vehicle import *  
from MARC.Visualization.Performance.Mission              import *  
from MARC.Visualization.Performance.Energy.Common        import *  
from MARC.Visualization.Performance.Energy.Fuel          import *   
from MARC.Visualization.Performance.Noise                import * 

def plot_mission(results): 
    
    plot_altitude_sfc_weight(results,show_figure=False) 
    
    plot_flight_conditions(results,show_figure=False) 
    
    plot_aerodynamic_coefficients(results,show_figure=False)  
    
    plot_aircraft_velocities(results,show_figure=False)
    
    plot_stability_coefficients(results,show_figure=False)
    
    plot_drag_components(results,show_figure=False)
 
    return
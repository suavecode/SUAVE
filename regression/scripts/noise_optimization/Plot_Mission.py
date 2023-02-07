# Plot_Mission.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------  
from MARC.Visualization.Performance.Aerodynamics.Vehicle import *  
from MARC.Visualization.Performance.Mission              import *  
from MARC.Visualization.Performance.Energy.Common        import *  
from MARC.Visualization.Performance.Energy.Battery       import *   
from MARC.Visualization.Performance.Energy.Fuel          import *  
from MARC.Visualization.Performance.Noise                import * 
from MARC.Visualization.Geometry.Three_Dimensional       import * 

import pylab as plt

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results):
    

    # Plot Flight Conditions 
    plot_flight_conditions(results,show_figure=False)
    
    # Plot Aerodynamic Forces 
    plot_aerodynamic_forces(results,show_figure=False)
    
    # Plot Aerodynamic Coefficients 
    plot_aerodynamic_coefficients(results,show_figure=False)
    
    # Plot Static Stability Coefficients 
    plot_stability_coefficients(results,show_figure=False)    
    
    # Drag Components
    plot_drag_components(results,show_figure=False)
    
    # Plot Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results,show_figure=False)
    
    # Plot Velocities 
    plot_aircraft_velocities(results,show_figure=False)   
     
    return

if __name__ == '__main__':  
    plt.show()
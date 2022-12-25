# Plot_Mission.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Aerodynamics.Vehicle import *  
from SUAVE.Visualization.Performance.Mission              import *  
from SUAVE.Visualization.Performance.Energy.Common        import *  
from SUAVE.Visualization.Performance.Energy.Battery       import *   
from SUAVE.Visualization.Performance.Energy.Fuel          import *  
from SUAVE.Visualization.Performance.Noise                import * 
from SUAVE.Visualization.Geometry.Three_Dimensional       import * 

import pylab as plt

# ----------------------------------------------------------------------
#   Plot Mission
# ----------------------------------------------------------------------

def plot_mission(results):
    

    # Plot Flight Conditions 
    plot_flight_conditions(results)
    
    # Plot Aerodynamic Forces 
    plot_aerodynamic_forces(results)
    
    # Plot Aerodynamic Coefficients 
    plot_aerodynamic_coefficients(results)
    
    # Plot Static Stability Coefficients 
    plot_stability_coefficients(results)    
    
    # Drag Components
    plot_drag_components(results)
    
    # Plot Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results)
    
    # Plot Velocities 
    plot_aircraft_velocities(results)   
     
    return

if __name__ == '__main__':  
    plt.show()
# Plot_Mission.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import MARC
from MARC.Core import Units
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
# Plot_Mission.py
# 
# Created:  May 2015, E. Botero
# Modified: 

# ----------------------------------------------------------------------        
#   Imports
# ----------------------------------------------------------------------    

import SUAVE
from SUAVE.Core import Units
from SUAVE.Plots.Mission_Plots import * 
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

    # Drag Components
    plot_drag_components(results)

    # Plot Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results)       
    
    return

if __name__ == '__main__': 
    plt.show()
# test_plots.py
# 
# Created: May 2019, M. Clarke
#
# Tests plotting functions 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units
from SUAVE.Plots.Mission_Plots import * 
import matplotlib.pyplot as plt  

def main():
    results = load_plt_data()
     
    # Compare Plot for  Aerodynamic Forces 
    plot_aerodynamic_forces(results)
    
    # Compare Plot for  Aerodynamic Coefficients 
    plot_aerodynamic_coefficients(results)
    
    # Compare Plot for Drag Components
    plot_drag_components(results)
    
    # Compare Plot for  Altitude, sfc, vehicle weight 
    plot_altitude_sfc_weight(results)
    
    # Compare Plot for Aircraft Velocities 
    plot_aircraft_velocities(results)      

    # Compare Plot for Flight Conditions   
    plot_flight_conditions(results)

    return 

def load_plt_data():
    return SUAVE.Input_Output.SUAVE.load('../B737/plot_data_B737.res')

if __name__ == '__main__':     
    main()  
    plt.show()
    print('Plots regression test passed!')
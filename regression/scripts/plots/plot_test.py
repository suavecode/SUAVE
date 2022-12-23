# test_plots.py
# 
# Created: Mar 2020, M. Clarke
# Modified: Jan 2022, S. Claridge
# Tests plotting functions 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import SUAVE 
from SUAVE.Visualization.Performance.Vehicle_Aerodynamics import *  
from SUAVE.Visualization.Performance.Mission import *  
from SUAVE.Visualization.Performance.Energy.Fuel import *   
from SUAVE.Visualization.Performance.Noise import *  
import matplotlib.pyplot as plt  

def main():
    """This test loads results from the B737 regression to test the plot functions 
    """
    results = load_plt_data()
    
    """
    # Compare Plot for  Aerodynamic Forces 
    """
    plot_aerodynamic_forces(results)
    
    
    """
    # Compare Plot for  Aerodynamic Coefficients 
    """
    plot_aerodynamic_coefficients(results) 
    
    
    """
    # Compare Plot for Drag Components
    """
    plot_drag_components(results)
    
    
    """
    # Compare Plot for  Altitude, sfc, vehicle weight 
    """
    plot_altitude_sfc_weight(results)
    
    
    """
    # Compare Plot for Aircraft Velocities 
    """
    plot_aircraft_velocities(results)      


    """
    # Compare Plot for Flight Conditions   
    """
    plot_flight_conditions(results)
    

    """
    # Compare Plot for Flight Trajectory
    """
    plot_flight_trajectory(results) 

    
    """
    # Compare Plot for Fuel Tracking 
    """
    plot_fuel_use(results)


    return 

def load_plt_data():
    return SUAVE.Input_Output.SUAVE.load('../B737/results_mission_B737.res')

if __name__ == '__main__':     
    main()  
    plt.show()
    print('Plots regression test passed!')
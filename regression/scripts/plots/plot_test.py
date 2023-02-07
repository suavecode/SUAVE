# test_plots.py
# 
# Created: Mar 2020, M. Clarke
# Modified: Jan 2022, S. Claridge
# Tests plotting functions 

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------
import MARC 
from MARC.Visualization.Performance.Aerodynamics.Vehicle import *  
from MARC.Visualization.Performance.Mission              import *  
from MARC.Visualization.Performance.Energy.Common        import *  
from MARC.Visualization.Performance.Energy.Battery       import *   
from MARC.Visualization.Performance.Energy.Fuel          import *  
from MARC.Visualization.Performance.Noise                import *    
import matplotlib.pyplot as plt  

def main():
    """This test loads results from the B737 regression to test the plot functions 
    """
    results = load_plt_data()
    
    """
    # Compare Plot for  Aerodynamic Forces 
    """
    plot_aerodynamic_forces(results,show_figure=False)
    
    
    """
    # Compare Plot for  Aerodynamic Coefficients 
    """
    plot_aerodynamic_coefficients(results,show_figure=False) 
    
    
    """
    # Compare Plot for Drag Components
    """
    plot_drag_components(results,show_figure=False)
    
    
    """
    # Compare Plot for  Altitude, sfc, vehicle weight 
    """
    plot_altitude_sfc_weight(results,show_figure=False)
    
    
    """
    # Compare Plot for Aircraft Velocities 
    """
    plot_aircraft_velocities(results,show_figure=False)      


    """
    # Compare Plot for Flight Conditions   
    """
    plot_flight_conditions(results,show_figure=False)
    

    """
    # Compare Plot for Flight Trajectory
    """
    plot_flight_trajectory(results,show_figure=False)  

    
    """
    # Compare Plot for Fuel Tracking 
    """
    plot_fuel_use(results,show_figure=False)


    return 

def load_plt_data():
    return MARC.Input_Output.MARC.load('../B737/results_mission_B737.res')

if __name__ == '__main__':     
    main()  
    plt.show()
    print('Plots regression test passed!')
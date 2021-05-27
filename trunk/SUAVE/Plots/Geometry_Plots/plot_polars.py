## @ingroup Plots-Geometry_Plots
# plot_polars.py
# 
# Created:  May 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import matplotlib.pyplot as plt   
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars

def plot_polars(airfoil_polar_paths, line_color = 'k-', raw_polars = True, overlay = True, 
                save_figure = False, save_filename = "Airfoil_Polars", file_type = ".png"):
    """This plots all airfoil polars in the list "airfoil_polar_paths" 

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_polar_paths   [list of strings]

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """
    if raw_polars:
        # Plot the raw data polars
        airfoil_polar_data = import_airfoil_polars(airfoil_polar_paths)
        CL = airfoil_polar_data.lift_coefficients
        CD = airfoil_polar_data.drag_coefficients
        
    else:
        # Plot surrogate polars
        plot_surrogate_polars(airfoil_polar_paths)
            
    
    if overlay:
        name = save_filename
        fig  = plt.figure(name)
        fig.set_size_inches(10, 4)
        axes = fig.add_subplot(1,1,1)
        for i in range(len(airfoil_polar_data)):
            polar_label = str(i)
            axes.plot(CL[i], CD[i] , label=polar_label )                  
        
        axes.set_title("Airfoil Polars")
        axes.legend()
        if save_figure:
            plt.savefig(name + file_type)   
            
    else:
        for i in range(len(airfoil_names)):
            # separate x and y coordinates 
            airfoil_x  = airfoil_data.x_coordinates[i] 
            airfoil_y  = airfoil_data.y_coordinates[i]    
    
            name = save_filename + '_' + str(i)
            fig  = plt.figure(name)
            axes = fig.add_subplot(1,1,1)
            axes.set_title(airfoil_names[i])
            axes.plot(airfoil_x, airfoil_y , line_color )                  
            axes.axis('equal')
            if save_figure:
                plt.savefig(name + file_type)          

    return
## @ingroup Plots-Geometry_Plots
# plot_airfoil.py
# 
# Created:    Mar 2020, M. Clarke
# Modified:   Apr 2020, M. Clarke
#             Jul 2020, M. Clarke
#             May 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import matplotlib.pyplot as plt   
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry 
import os

def plot_airfoil(airfoil_paths,  line_color = 'k-', overlay = False, save_figure = False, save_filename = "Airfoil_Geometry", file_type = ".png"):
    """This plots all airfoil defined in the list "airfoil_names" 

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_geometry_files   <list of strings>

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """
    # get airfoil coordinate geometry     
    airfoil_data = import_airfoil_geometry(airfoil_paths)
    
    if overlay:
        name = save_filename
        fig  = plt.figure(name)
        fig.set_size_inches(10, 4)
        axes = fig.add_subplot(1,1,1)
        for i in range(len(airfoil_paths)):
            # extract airfoil name from path
            airfoil_name = os.path.basename(airfoil_paths[i])
            
            # separate x and y coordinates 
            airfoil_x  = airfoil_data.x_coordinates[i] 
            airfoil_y  = airfoil_data.y_coordinates[i]    
            
            # plot airfoil geometry
            axes.plot(airfoil_x, airfoil_y , label=airfoil_name )                  
        
        axes.set_title("Airfoil Geometry")
        axes.legend(bbox_to_anchor=(1,1), loc='upper left', ncol=1)
        if save_figure:
            plt.savefig(name + file_type)   
            
    else:
        for i in range(len(airfoil_paths)):
            # extract airfoil name from path
            airfoil_name = os.path.basename(airfoil_paths[i])
            
            # separate x and y coordinates 
            airfoil_x  = airfoil_data.x_coordinates[i] 
            airfoil_y  = airfoil_data.y_coordinates[i]    
    
            name = save_filename + '_' + str(i)
            fig  = plt.figure(name)
            axes = fig.add_subplot(1,1,1)
            axes.set_title(airfoil_name)
            axes.plot(airfoil_x, airfoil_y , line_color )                  
            axes.axis('equal')
            if save_figure:
                plt.savefig(name + file_type)          

    return

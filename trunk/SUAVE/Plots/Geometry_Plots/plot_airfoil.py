## @ingroup Plots-Geometry_Plots
# plot_airfoil.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import matplotlib.pyplot as plt   
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry 

## @ingroup Plots-Geometry_Plots
def plot_airfoil(airfoil_names,  line_color = 'k-', save_figure = False, save_filename = "Airfoil_Geometry", file_type = ".png"):
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
    airfoil_data = import_airfoil_geometry(airfoil_names)       

    for i in range(len(airfoil_names)):
        # separate x and y coordinates 
        airfoil_x  = airfoil_data.x_coordinates[i] 
        airfoil_y  = airfoil_data.y_coordinates[i]    

        name = save_filename + '_' + str(i)
        fig  = plt.figure(name)
        axes = fig.add_subplot(1,1,1)
        axes.set_title(airfoil_names[i])
        axes.plot(airfoil_x, airfoil_y , line_color )                  
        #axes.set_aspect('equal')
        axes.axis('equal')
        if save_figure:
            plt.savefig(name + file_type)          

    return

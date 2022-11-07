## @ingroup Plots-Geometry
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

def plot_airfoil(airfoil_paths,line_color = 'k-', save_figure = False, save_filename = "Airfoil_Geometry", file_type = ".png"):
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
    airfoil_geometry = import_airfoil_geometry(airfoil_paths)
     
    fig  = plt.figure(save_filename)
    fig.set_size_inches(10, 4)
    axes = fig.add_subplot(1,1,1)   
    axes.plot(airfoil_geometry.x_coordinates,airfoil_geometry.y_coordinates , label=save_filename)                  
    
    axes.set_title(save_filename) 
    if save_figure:
        plt.savefig(save_filename.replace("_", " ") + file_type)           

    return

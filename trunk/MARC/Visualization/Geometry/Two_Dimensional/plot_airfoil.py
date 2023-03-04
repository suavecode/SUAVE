## @ingroup Visualization-Geometry
# plot_airfoil.py
# 
# Created:    Mar 2020, M. Clarke
# Modified:   Apr 2020, M. Clarke
#             Jul 2020, M. Clarke
#             May 2021, R. Erhard
#             Nov 2022, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
from MARC.Visualization.Performance.Common import plot_style
import matplotlib.pyplot as plt 
from MARC.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry 

## @ingroup Visualization-Geometry
def plot_airfoil(airfoil_paths,
                 save_figure = False, 
                 save_filename = "Airfoil_Geometry",
                 file_type = ".png", 
                 width = 12, height = 7):
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

    # get plotting style 
    ps      = plot_style()  

    parameters = {'axes.labelsize': ps.axis_font_size,
                  'xtick.labelsize': ps.axis_font_size,
                  'ytick.labelsize': ps.axis_font_size,
                  'axes.titlesize': ps.title_font_size}
    plt.rcParams.update(parameters)    

    fig  = plt.figure(save_filename)
    fig.set_size_inches(width,height) 
    axis = fig.add_subplot(1,1,1)     
    axis.plot(airfoil_geometry.x_coordinates,airfoil_geometry.y_coordinates, color = ps.color, marker = ps.marker, linewidth = ps.line_width) 
    axis.set_xlabel('x')
    axis.set_ylabel('y')    
     
    if save_figure:
        fig.savefig(save_filename.replace("_", " ") + file_type)  
     
    return

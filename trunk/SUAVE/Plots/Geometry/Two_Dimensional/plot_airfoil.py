## @ingroup Plots-Geometry
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
import plotly.express as px
import pandas as pd
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry \
     import import_airfoil_geometry 

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
    
    df = pd.DataFrame(dict(x=airfoil_geometry.x_coordinates, y=airfoil_geometry.y_coordinates))
    fig = px.line(df, x='x', y='y',title=save_filename)
    fig.update_layout(autosize=False, width=1200, height=500)
    
    if save_figure:
        fig.write_image(save_filename.replace("_", " ") + file_type)  
        
    fig.show()

    return

## @ingroup Visualization-Geometry-Three_Dimensional
# plot_3d_wing.py
#
# Created : Mar 2020, M. Clarke
# Modified: Apr 2020, M. Clarke
# Modified: Jul 2020, M. Clarke
# Modified: Jul 2021, E. Botero
# Modified: Oct 2021, M. Clarke
# Modified: Dec 2021, M. Clarke
# Modified: Feb 2022, R. Erhard
# Modified: Mar 2022, R. Erhard
# Modified: Sep 2022, M. Clarke
# Modified: Nov 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import numpy as np     
from SUAVE.Visualization.Geometry.Common.contour_surface_slice import contour_surface_slice

## @ingroup Visualization-Geometry-Three_Dimensional
def plot_3d_wing(plot_data,VD,color_map = 'greys',values = None, angle_of_attack = 0):
    """ This plots the wings of a vehicle

    Assumptions:
    None

    Source:
    None

    Inputs:
    VD.
       XA1...ZB2    - coordinates of wing vortex distribution
    color_map       - color of panel 

    Properties Used:
    N/A
    """
    
    if values is None:
        monocolor = True 
    else: 
        monocolor = False
        
    n_cp = VD.n_cp
    for i in range(n_cp): 

        X = np.array([[VD.XA1[i],VD.XA2[i]],[VD.XB1[i],VD.XB2[i]]])
        Y = np.array([[VD.YA1[i],VD.YA2[i]],[VD.YB1[i],VD.YB2[i]]])
        Z = np.array([[VD.ZA1[i],VD.ZA2[i]],[VD.ZB1[i],VD.ZB2[i]]])   
        
        if monocolor:
            values = np.ones_like(X)
            
        verts = contour_surface_slice(X, Y, Z ,values,color_map)
        plot_data.append(verts)            

    return plot_data  
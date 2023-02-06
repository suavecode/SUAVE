## @ingroup Visualization-Geometry-Three_Dimensional 
# plot_3d_fuselage.py
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
from MARC.Core import Data
import numpy as np  
import plotly.graph_objects as go 
from MARC.Visualization.Geometry.Common.contour_surface_slice import contour_surface_slice

## @ingroup Visualization-Geometry-Three_Dimensional 
def plot_3d_fuselage(plot_data,fus, tessellation = 24 ,color_map = 'teal'):
    """ This plots the 3D surface of the fuselage

    Assumptions:
    None

    Source:
    None

    Inputs:
    fus                  - fuselage data structure
    fus_pts              - coordinates of fuselage points
    color_map            - color of panel 


    Properties Used:
    N/A
    """
    fus_pts      = generate_3d_fuselage_points(fus,tessellation = 24 )
    num_fus_segs = len(fus_pts[:,0,0])
    if num_fus_segs > 0:
        tesselation  = len(fus_pts[0,:,0])
        for i_seg in range(num_fus_segs-1):
            for i_tes in range(tesselation-1):
                X = np.array([[fus_pts[i_seg  ,i_tes,0],fus_pts[i_seg+1,i_tes  ,0]],
                              [fus_pts[i_seg  ,i_tes+1,0],fus_pts[i_seg+1,i_tes+1,0]]])
                Y = np.array([[fus_pts[i_seg  ,i_tes  ,1],fus_pts[i_seg+1,i_tes  ,1]],
                              [fus_pts[i_seg  ,i_tes+1,1],fus_pts[i_seg+1,i_tes+1,1]]])
                Z = np.array([[fus_pts[i_seg  ,i_tes  ,2],fus_pts[i_seg+1,i_tes  ,2]],
                              [fus_pts[i_seg  ,i_tes+1,2],fus_pts[i_seg+1,i_tes+1,2]]])  
                 
                values = np.ones_like(X) 
                verts  = contour_surface_slice(X, Y, Z,values,color_map)
                plot_data.append(verts)          

    return plot_data 

def generate_3d_fuselage_points(fus ,tessellation = 24 ):
    """ This generates the coordinate points on the surface of the fuselage

    Assumptions:
    None

    Source:
    None

    Inputs:
    fus                  - fuselage data structure

    Properties Used:
    N/A
    """
    num_fus_segs = len(fus.Segments.keys())
    fus_pts      = np.zeros((num_fus_segs,tessellation ,3))

    if num_fus_segs > 0:
        for i_seg in range(num_fus_segs):
            theta    = np.linspace(0,2*np.pi,tessellation)
            a        = fus.Segments[i_seg].width/2
            b        = fus.Segments[i_seg].height/2
            r        = np.sqrt((b*np.sin(theta))**2  + (a*np.cos(theta))**2)
            fus_ypts = r*np.cos(theta)
            fus_zpts = r*np.sin(theta)
            fus_pts[i_seg,:,0] = fus.Segments[i_seg].percent_x_location*fus.lengths.total + fus.origin[0][0]
            fus_pts[i_seg,:,1] = fus_ypts + fus.Segments[i_seg].percent_y_location*fus.lengths.total + fus.origin[0][1]
            fus_pts[i_seg,:,2] = fus_zpts + fus.Segments[i_seg].percent_z_location*fus.lengths.total + fus.origin[0][2]

    return fus_pts

## @ingroup Visualization-Geometry
# plot_3d_vehicle_vlm_panelization.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke
#           Dec 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
import numpy as np  
import plotly.graph_objects as go 
from SUAVE.Visualization.Geometry.Common.contour_surface_slice import contour_surface_slice

## @ingroup Visualization-Geometry
def plot_3d_vehicle_vlm_panelization(vehicle, alpha = 1.0 ,plot_axis = False,
                                  save_figure = False, plot_wing_control_points = True, save_filename = "VLM_Panelization"):
                                  
    """This plots vortex lattice panels created when Fidelity Zero  Aerodynamics 
    Routine is initialized

    Assumptions:
    None

    Source:
    None

    Inputs:
    vehicle.vortex_distribution

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """
    # unpack vortex distribution 
    VD = vehicle.vortex_distribution

    camera        = dict(up=dict(x=0.5, y=0.5, z=1), center=dict(x=0, y=0, z=-0.5), eye=dict(x=-1.5, y=-1.5, z=.8))
    plot_data     = []     
     
    
    # -------------------------------------------------------------------------
    # DEFINE PLOT LIMITS 
    # -------------------------------------------------------------------------    
    y_min,y_max = np.min(VD.YC)*1.2, np.max(VD.YC)*1.2
    x_min,x_max = np.minimum(0,np.min(VD.XC)*1.2), np.maximum(np.max(VD.XC)*1.2, 2*y_max)
    z_min,z_max = -np.max(VD.ZC)*1.2, np.max(VD.ZC)*1.2

    # -------------------------------------------------------------------------
    # PLOT VORTEX LATTICE
    # -------------------------------------------------------------------------        
    n_cp      = VD.n_cp 
    color_map = 'greys'
    for i in range(n_cp):  
        X = np.array([[VD.XA1[i],VD.XA2[i]],[VD.XB1[i],VD.XB2[i]]])
        Y = np.array([[VD.YA1[i],VD.YA2[i]],[VD.YB1[i],VD.YB2[i]]])
        Z = np.array([[VD.ZA1[i],VD.ZA2[i]],[VD.ZB1[i],VD.ZB2[i]]])           
        
        values      = np.ones_like(X) 
        verts       = contour_surface_slice(X, Y, Z ,values,color_map)
        plot_data.append(verts)                 
  
  
    if  plot_wing_control_points: 
        ctrl_pts = go.Scatter3d(x=VD.XC, y=VD.YC, z=VD.ZC,
                                    mode  = 'markers',
                                    marker= dict(size=6,color='red',opacity=0.8),
                                    line  = dict(color='red',width=2))
        plot_data.append(ctrl_pts)         
 
 
 
    fig = go.Figure(data=plot_data)
    fig.update_scenes(aspectmode   = 'auto',
                      xaxis_visible=plot_axis,
                      yaxis_visible=plot_axis,
                      zaxis_visible=plot_axis
                      )
    fig.update_layout( 
             width     = 1500,
             height    = 1500, 
             scene = dict(
                        xaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis,
                                     zerolinecolor="white", range=[x_min,x_max]),
                        yaxis = dict(backgroundcolor="grey", gridcolor="white", showbackground=plot_axis, 
                                     zerolinecolor="white", range=[y_min,y_max]),
                        zaxis = dict(backgroundcolor="grey",gridcolor="white",showbackground=plot_axis,
                                     zerolinecolor="white", range=[z_min,z_max])),             
             scene_camera=camera) 
    fig.update_coloraxes(showscale=False)
    fig.update_traces(opacity = alpha)
    if save_figure:
        fig.write_image(save_filename + ".png")
    fig.show()     
    return 

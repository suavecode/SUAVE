## @ingroup Plots-Geometry
# plot_vehicle_vlm_panelization.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke
#           Dec 2021, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
import numpy as np 
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

## @ingroup Plots-Geometry
def plot_vehicle_vlm_panelization(vehicle, elevation_angle = 30,azimuthal_angle = 210, axis_limits = 10,plot_axis = False,
                                  save_figure = False, plot_control_points = True, save_filename = "VLM_Panelization"):
                                  
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
    
    face_color = 'grey'        
    edge_color = 'black'
    alpha_val  = 0.5  
    
    # initalize figure 
    fig = plt.figure(save_filename)
    fig.set_size_inches(8,8)
    axes = plt.axes(projection='3d')
    axes.view_init(elev= elevation_angle, azim= azimuthal_angle) 
    
    n_cp = VD.n_cp 
    for i in range(n_cp): 
        X = [VD.XA1[i],VD.XB1[i],VD.XB2[i],VD.XA2[i]]
        Y = [VD.YA1[i],VD.YB1[i],VD.YB2[i],VD.YA2[i]]
        Z = [VD.ZA1[i],VD.ZB1[i],VD.ZB2[i],VD.ZA2[i]] 
        verts = [list(zip(X, Y, Z))]
        collection = Poly3DCollection(verts)
        collection.set_facecolor(face_color)
        collection.set_edgecolor(edge_color)
        collection.set_alpha(alpha_val)
        axes.add_collection3d(collection)                
  
  
    if  plot_control_points:
        axes.scatter(VD.XC,VD.YC,VD.ZC, c='r', marker = 'o' )  

    axes.set_xlim(0,axis_limits*2)
    axes.set_ylim(-axis_limits,axis_limits)
    axes.set_zlim(-axis_limits,axis_limits)
    
    if not plot_axis:
        plt.axis('off')
        plt.grid(None)       
    return 

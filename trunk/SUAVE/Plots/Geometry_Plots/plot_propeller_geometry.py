## @ingroup Plots-Geometry_Plots
# plot_propeller_geometry.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
import matplotlib.pyplot as plt   
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

## @ingroup Plots-Geometry_Plots
def plot_propeller_geometry(prop, face_color = 'gray', edge_color = 'black' , save_figure = False, save_filename = "Propeller_Geometry", file_type = ".png"):
    """This plots the geoemtry of a propeller or rotor

    Assumptions:
    None

    Source:
    None

    Inputs:
    SUAVE.Components.Energy.Converters.Propeller()

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """	
    # unpack
    G = prop.geometry_panelization
    fig = plt.figure(save_filename)
    fig.set_size_inches(10, 8)     
    axes = Axes3D(fig) 
    axes.view_init(elev=10., azim=10)             
    axes.set_zlim3d(-1, 1)        
    axes.set_ylim3d(-1, 1)        
    axes.set_xlim3d(-1, 1)     
    plt.axis('off') 
    plt.grid(None)   
    alpha      = 1
    
    num_prop = len(G.XA1[:,0,0,0])
    num_B    = len(G.XA1[0,:,0,0])
    num_sec  = len(G.XA1[0,0,:,0])
    num_surf = len(G.XA1[0,0,0,:])
    for p_idx in range(num_prop):  
            for B_idx in range(num_B):
                for sec in range(num_sec): 
                    for loc in range(num_surf): 
                        X = [G.XA1[p_idx,B_idx,sec,loc],
                             G.XB1[p_idx,B_idx,sec,loc],
                             G.XB2[p_idx,B_idx,sec,loc],
                             G.XA2[p_idx,B_idx,sec,loc]]
                        Y = [G.YA1[p_idx,B_idx,sec,loc],
                             G.YB1[p_idx,B_idx,sec,loc],
                             G.YB2[p_idx,B_idx,sec,loc],
                             G.YA2[p_idx,B_idx,sec,loc]]
                        Z = [G.ZA1[p_idx,B_idx,sec,loc],
                             G.ZB1[p_idx,B_idx,sec,loc],
                             G.ZB2[p_idx,B_idx,sec,loc],
                             G.ZA2[p_idx,B_idx,sec,loc]]                    
                        verts = [list(zip(X, Y, Z))]
                        collection = Poly3DCollection(verts)
                        collection.set_facecolor(face_color)
                        collection.set_edgecolor(edge_color) 
                        collection.set_alpha(alpha)
                        axes.add_collection3d(collection)     

    if save_figure:
        plt.savefig(save_filename + file_type)  

    return 

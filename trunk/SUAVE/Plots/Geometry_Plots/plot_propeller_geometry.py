## @ingroup Plots-Geometry_Plots
# plot_propeller_geometry.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
from SUAVE.Core import Units
import matplotlib.pyplot as plt   
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from SUAVE.Plots.Geometry_Plots import generate_propeller_geometry

## @ingroup Plots-Geometry_Plots
def plot_propeller_geometry(prop, face_color = 'red', edge_color = 'black' , save_figure = False, save_filename = "Propeller_Geometry", file_type = ".png"):
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
    G    = generate_propeller_geometry(prop)    
    
    fig = plt.figure(save_filename + '_3D' )
    fig.set_size_inches(10, 8)     
    axes =  fig.add_subplot(111, projection='3d')
    axes.set_zlim3d(-1, 1)        
    axes.set_ylim3d(-1, 1)        
    axes.set_xlim3d(-1, 1)   
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
                        prop_verts = [list(zip(X, Y, Z))]
                        prop_collection = Poly3DCollection(prop_verts)
                        prop_collection.set_facecolor(face_color[p_idx])
                        prop_collection.set_edgecolor(edge_color[p_idx]) 
                        prop_collection.set_alpha(alpha)
                        axes.add_collection3d(prop_collection)    

    if save_figure:
        plt.savefig(save_filename + '_3D' + file_type)  
        
    fig = plt.figure(save_filename)
    fig.set_size_inches(12, 8)    
    axes = fig.add_subplot(2,2,1)
    axes.plot(prop.radius_distribution, prop.twist_distribution/Units.degrees,'bo-')  
    axes.set_ylabel('Twist (Deg)') 
    axes.set_xlabel('Radial Station')    
    
    axes = fig.add_subplot(2,2,2)
    axes.plot(prop.radius_distribution, prop.chord_distribution,'bo-')    
    axes.set_ylabel('Chord (m)') 
    axes.set_xlabel('Radial Station')    
    
    axes = fig.add_subplot(2,2,3)
    axes.plot(prop.radius_distribution  , prop.max_thickness_distribution,'bo-')     
    axes.set_ylabel('Thickness (m)')  
    axes.set_xlabel('Radial Station')    
    
    axes = fig.add_subplot(2,2,4)
    axes.plot(prop.radius_distribution  , prop.mid_chord_aligment,'bo-')  
    axes.set_ylabel('Mid Chord Alignment (m)')  
    axes.set_xlabel('Radial Station')    
    
    if save_figure:
        plt.savefig(save_filename + '_3D' + file_type)  
        
    return 

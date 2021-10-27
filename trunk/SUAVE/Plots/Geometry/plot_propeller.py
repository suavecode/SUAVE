## @ingroup Plots-Geometry
# plot_propeller.py
# 
# Created:  Mar 2020, M. Clarke
#           Apr 2020, M. Clarke
#           Jul 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
from SUAVE.Core import Units
import matplotlib.pyplot as plt    
from SUAVE.Plots.Geometry.plot_vehicle import plot_propeller_geometry
from SUAVE.Components.Energy.Networks.Battery_Propeller import Battery_Propeller

## @ingroup Plots-Geometry
def plot_propeller(prop, face_color = 'red', edge_color = 'black' , save_figure = False, save_filename = "Propeller_Geometry", file_type = ".png"):
    """This plots the geometry of a propeller or rotor

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
    
    # initalize figure 
    fig_1 = plt.figure(save_filename + '_3D') 
    fig_1.set_size_inches(8,8) 
    axes_1 = plt.axes(projection='3d')
    axes_1.view_init(elev= 30, azim= 210)   
    axes_1.set_xlim(-1,1)
    axes_1.set_ylim(-1,1)
    axes_1.set_zlim(-1,1) 
    axes_1.set_xlabel('x')  
    axes_1.set_ylabel('y')   
    axes_1.set_zlabel('z')    
    
    # append a network for origin and thrust angle default values
    network = Battery_Propeller() 
    
    # plot propeller geometry
    plot_propeller_geometry(axes_1,prop,network,prop.tag) 
    
    if save_figure:
        plt.savefig(save_filename + '_3D' + file_type)  
        
    fig_2 = plt.figure(save_filename + '_2D')
    fig_2.set_size_inches(12, 8)    
    axes_1 = fig_2.add_subplot(2,2,1)
    axes_1.plot(prop.radius_distribution, prop.twist_distribution/Units.degrees,'bo-')  
    axes_1.set_ylabel('Twist (Deg)') 
    axes_1.set_xlabel('Radial Station')    
    
    axes_2 = fig_2.add_subplot(2,2,2)
    axes_2.plot(prop.radius_distribution, prop.chord_distribution,'bo-')    
    axes_2.set_ylabel('Chord (m)') 
    axes_2.set_xlabel('Radial Station')    
    
    axes_3 = fig_2.add_subplot(2,2,3)
    axes_3.plot(prop.radius_distribution  , prop.max_thickness_distribution,'bo-')     
    axes_3.set_ylabel('Thickness (m)')  
    axes_3.set_xlabel('Radial Station')    
    
    axes_4 = fig_2.add_subplot(2,2,4)
    axes_4.plot(prop.radius_distribution  , prop.mid_chord_alignment,'bo-')  
    axes_4.set_ylabel('Mid Chord Alignment (m)')  
    axes_4.set_xlabel('Radial Station')    
    
    if save_figure:
        plt.savefig(save_filename + '_2D' + file_type)  
        
    return 

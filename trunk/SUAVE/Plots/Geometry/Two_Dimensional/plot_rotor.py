## @ingroup Plots-Geometry
# plot_2d_rotor.py
# 
# Created:  Mar 2020, M. Clarke
# Modified: Apr 2020, M. Clarke
#           Jul 2020, M. Clarke
#           Feb 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
from SUAVE.Core import Units
import matplotlib.pyplot as plt      

## @ingroup Plots-Geometry
def plot_rotor(prop, face_color = 'red', edge_color = 'black' , save_figure = False, save_filename = "Propeller_Geometry", file_type = ".png"):
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

## @ingroup Visualization-Performance-Mission
# plot_flight_trajectory.py
# 
# Created:    Dec 2022, M. Clarke
# Modified:   

from MARC.Core import Units
from MARC.Visualization.Performance.Common import set_axes, plot_style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np  

# ------------------------------------------------------------------
#  2D Aircraft Trajectory
# ------------------------------------------------------------------
## @ingroup Visualization-Performance-Mission          
def plot_flight_trajectory(results,
                           line_color = 'bo-',
                           line_color2 = 'rs--',
                           save_figure = False,
                           save_filename = "Flight_Trajectory",
                           file_type = ".png",
                           width = 12, height = 7):
    """This plots the 3D flight trajectory of the aircraft.

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.
         frames 
             body.inertial_rotations
             inertial.position_vector 
         freestream.velocity
         aerodynamics.
             lift_coefficient
             drag_coefficient
             angle_of_attack
        
    Outputs: 
    Plots

    Properties Used:
    N/A	
    """	     

    # get plotting style 
    ps      = plot_style()  

    parameters = {'axes.labelsize': ps.axis_font_size,
                  'xtick.labelsize': ps.axis_font_size,
                  'ytick.labelsize': ps.axis_font_size,
                  'axes.titlesize': ps.title_font_size}
    plt.rcParams.update(parameters) 
        
    fig = plt.figure(save_filename)
    fig.set_size_inches(width,height) 
     
    # get line colors for plots 
    line_colors   = cm.inferno(np.linspace(0,0.9,len(results.segments)))    
     
    for i in range(len(results.segments)): 
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        x        = results.segments[i].conditions.frames.inertial.position_vector[:,0] 
        y        = results.segments[i].conditions.frames.inertial.position_vector[:,1] 
        z        = -results.segments[i].conditions.frames.inertial.position_vector[:,2] 
        
        axes = plt.subplot(2,2,1)
        axes.plot( time , x/1000, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width )
        axes.set_xlabel('Distance (km)')
        axes.set_xlabel('Time (min)')
        set_axes(axes)            

        axes = plt.subplot(2,2,2)
        axes.plot(x, y , line_color, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width )
        axes.set_xlabel('x (m)')
        axes.set_ylabel('y (m)')
        set_axes(axes)

        axes = plt.subplot(2,2,3)
        axes.plot( time , z, line_color , color = line_colors[i], marker = ps.marker, linewidth = ps.line_width )
        axes.set_ylabel('z (m)')
        axes.set_xlabel('Time (min)')
        set_axes(axes)   
        
        axes = plt.subplot(2,2,4, projection='3d') 
        axes.scatter(x, y, z, marker='o',color =  line_colors[i])
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z') 
        set_axes(axes)         
    
    plt.tight_layout()    
    if save_figure:
        plt.savefig(save_filename + file_type)
        
    return             
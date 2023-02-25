## @defgroup Visualization-Performance-Mission 
# plot_flight_conditions.py
# 
# Created:    Dec 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Units
from MARC.Visualization.Performance.Common import set_axes, plot_style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

# ---------------------------------------------------------------------- 
#   Flight Condition
# ---------------------------------------------------------------------- 
## @defgroup Visualization-Performance-Mission 
def plot_flight_conditions(results,
                             save_figure = False,
                             show_legend=True,
                             save_filename = "Flight Conditions",
                             file_type = ".png",
                             width = 12, height = 7): 

    """This plots the flights the conditions

    Assumptions:
    None

    Source:
    
    Deprecated MARC Mission Plots Functions

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge
    

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
     
    # get line colors for plots 
    line_colors   = cm.inferno(np.linspace(0,0.9,len(results.segments)))     
     
    fig   = plt.figure()
    fig.set_size_inches(width,height)
    
    for i in range(len(results.segments)): 
        time     = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        airspeed = results.segments[i].conditions.freestream.velocity[:,0] /   Units['mph']
        theta    = results.segments[i].conditions.frames.body.inertial_rotations[:,1,None] / Units.deg
        Range    = results.segments[i].conditions.frames.inertial.aircraft_range[:,0]/ Units.nmi
        altitude = results.segments[i].conditions.freestream.altitude[:,0]/Units.feet
              
        segment_tag  =  results.segments[i].tag
        segment_name = segment_tag.replace('_', ' ')
        axes_1 = plt.subplot(2,2,1)
        axes_1.plot(time, altitude, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)
        axes_1.set_ylabel(r'Altitude (ft)')
        set_axes(axes_1)    
        
        axes_2 = plt.subplot(2,2,2)
        axes_2.plot(time, airspeed, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width) 
        axes_2.set_ylabel(r'Airspeed (mph)')
        set_axes(axes_2) 


        axes_3 = plt.subplot(2,2,3)
        axes_3.plot(time, Range, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_3.set_xlabel('Time (mins)')
        axes_3.set_ylabel(r'Range (nmi)')
        set_axes(axes_3) 
        
        axes_4 = plt.subplot(2,2,4)
        axes_4.plot(time, theta, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_4.set_xlabel('Time (mins)')
        axes_4.set_ylabel(r'Pitch Angle (deg)')
        set_axes(axes_4) 
         
    
    if show_legend:        
        leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
        leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
    
    # Adjusting the sub-plots for legend 
    fig.subplots_adjust(top=0.8)
    
    # set title of plot 
    title_text    = 'Flight Conditions'      
    fig.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return
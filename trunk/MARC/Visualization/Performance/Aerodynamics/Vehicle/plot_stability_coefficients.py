## @ingroup Visualization-Performance-Aerodynamics
# plot_stability_coefficients.py
# 
# Created:    Nov 2022, E. Botero
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
#   Stability Coefficients
# ---------------------------------------------------------------------- 

## @ingroup Visualization-Performance-Aerodynamics
def plot_stability_coefficients(results,
                             save_figure = False,
                             show_legend=True,
                             save_filename = "Stability_Coefficents",
                             file_type = ".png",
                             width = 12, height = 7):
    """This plots the static stability characteristics of an aircraft
    
    Assumptions:
    None
    
    Source:

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge
    
    
    Inputs:
    results.segments.conditions.stability.
       static
           CM
           Cm_alpha
           static_margin
       aerodynamics.
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
        c_m      = results.segments[i].conditions.stability.static.CM[:,0]
        cm_alpha = results.segments[i].conditions.stability.static.Cm_alpha[:,0]
        SM       = results.segments[i].conditions.stability.static.static_margin[:,0]
        aoa      = results.segments[i].conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
          
        segment_tag  =  results.segments[i].tag
        segment_name = segment_tag.replace('_', ' ')
        
        axes_1 = plt.subplot(2,2,1)
        axes_1.plot(time, aoa, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)
        axes_1.set_ylabel(r'AoA (deg)')
        set_axes(axes_1)    
        
        axes_2 = plt.subplot(2,2,2)
        axes_2.plot(time, c_m, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width) 
        axes_2.set_ylabel(r'$C_M$')
        set_axes(axes_2) 

        axes_3 = plt.subplot(2,2,3)
        axes_3.plot(time, cm_alpha, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_3.set_xlabel('Time (mins)')
        axes_3.set_ylabel(r'$C_M \alpha$')
        set_axes(axes_3) 
        
        axes_4 = plt.subplot(2,2,4)
        axes_4.plot(time,SM , color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_4.set_xlabel('Time (mins)')
        axes_4.set_ylabel(r'Static Margin (%)')
        set_axes(axes_4) 
        
    
    if show_legend:
        leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
        leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
    
    # Adjusting the sub-plots for legend 
    fig.subplots_adjust(top=0.8)
    
    # set title of plot 
    title_text    = 'Stability Coefficents'      
    fig.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return
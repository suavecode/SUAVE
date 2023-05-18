## @ingroup Visualization-Performance-Energy-Battery
# plot_battery_health_conditions.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Units
from MARC.Visualization.Performance.Common import set_axes, plot_style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

## @ingroup Visualization-Performance-Energy-Battery
def plot_battery_health_conditions(results,
                                  save_figure = False,
                                  show_legend = True,
                                  save_filename = "Battery_Health",
                                  file_type = ".png",
                                  width = 12, height = 7):
    """Plots the cell-level conditions of the battery throughout flight.

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
        freestream.altitude
        weights.total_mass
        weights.vehicle_mass_rate
        frames.body.thrust_force_vector

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
    
    fig   = plt.figure(save_filename)
    fig.set_size_inches(width,height) 
    for i in range(len(results.segments)): 
        time                = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min  
        cell_temp           = results.segments[i].conditions.propulsion.battery.cell.temperature[:,0]
        cell_charge         = results.segments[i].conditions.propulsion.battery.cell.charge_throughput[:,0] 

        segment_tag  =  results.segments[i].tag
        segment_name = segment_tag.replace('_', ' ')  

        axes_1 = plt.subplot(2,1,1)
        axes_1.plot(time, cell_charge, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)
        axes_1.set_xlabel('Time (mins)')
        axes_1.set_ylabel(r'Charge Throughput (Ah)')
        set_axes(axes_1)  

        axes_2 = plt.subplot(2,1,2)
        axes_2.plot(time,cell_temp, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_2.set_ylabel(r'Temperature (K)')
        axes_2.set_xlabel('Time (mins)')
        set_axes(axes_2)     
                
    
    if show_legend:        
        leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
        leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
    
    # Adjusting the sub-plots for legend 
    fig.subplots_adjust(top=0.75)
    
    # set title of plot 
    title_text    = 'Battery Health'      
    fig.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return
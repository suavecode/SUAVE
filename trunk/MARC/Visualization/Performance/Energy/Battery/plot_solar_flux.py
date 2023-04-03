## @ingroup Visualization-Performance-Energy-Battery
# plot_solar_flux.py
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
def plot_solar_flux(results,
                    save_figure = False,
                    show_legend=True,
                    save_filename = "Solar_Flux",
                    file_type = ".png",
                    width = 12, height = 7):
    """This plots the solar flux and power train performance of an solar powered aircraft

    Assumptions:
    None
    
    Deprecated MARC Mission Plots Functions 
    
    Inputs:
    results.segments.conditions.propulsion
        solar_flux
        battery_power_draw
        battery_energy
    
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

        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min
        flux   = results.segments[i].conditions.propulsion.solar_flux[:,0]
        charge = results.segments[i].conditions.propulsion.battery.pack.power_draw[:,0]
        current= results.segments[i].conditions.propulsion.battery.pack.current[:,0]
        energy = results.segments[i].conditions.propulsion.battery.pack.energy[:,0] / Units.MJ
        
                       
        segment_tag  =  results.segments[i].tag
        segment_name = segment_tag.replace('_', ' ')
        axes_1 = plt.subplot(2,2,1)
        axes_1.plot(time, flux, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)
        axes_1.set_ylabel(r'Solar Flux (W/m^2)')
        set_axes(axes_1)    
        
        axes_2 = plt.subplot(2,2,2)
        axes_2.plot(time, charge, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width) 
        axes_2.set_ylabel(r'Charging Power (W)')
        set_axes(axes_2) 

        axes_3 = plt.subplot(2,2,3)
        axes_3.plot(time, current, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_3.set_xlabel('Time (mins)')
        axes_3.set_ylabel(r'Battery Current (A)')
        set_axes(axes_3) 
        
        axes_4 = plt.subplot(2,2,4)
        axes_4.plot(time, energy, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_4.set_xlabel('Time (mins)')
        axes_4.set_ylabel(r'Battery Energy (MJ)')
        set_axes(axes_4) 
    
    if show_legend:        
        leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
        leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
    
    # Adjusting the sub-plots for legend 
    fig.subplots_adjust(top=0.8)
    
    # set title of plot 
    title_text    = 'Solar Flux Conditions'      
    fig.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return
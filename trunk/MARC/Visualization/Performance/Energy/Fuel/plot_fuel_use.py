## @ingroup Visualization-Performance-Energy-Fuel
# plot_fuel_use.py
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

## @ingroup Visualization-Performance-Energy-Fuel
def plot_fuel_use(results,
                    save_figure = False,
                    show_legend=True,
                    save_filename = "Aircraft_Fuel_Burnt",
                    file_type = ".png",
                    width = 12, height = 7): 

    """This plots aircraft fuel usage
    
    
    Assumptions:
    None

    Source:

    Depricated MARC Mission Plots Functions

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge

  
    Inputs:
    results.segments.condtions.
        frames.inertial.time
        weights.fuel_mass
        weights.additional_fuel_mass
        weights.total_mass
        
    Outputs:
    Plots
    Properties Used:
    N/A	"""

    # get plotting style 
    ps      = plot_style()  

    parameters = {'axes.labelsize': ps.axis_font_size,
                  'xtick.labelsize': ps.axis_font_size,
                  'ytick.labelsize': ps.axis_font_size,
                  'axes.titlesize': ps.title_font_size}
    plt.rcParams.update(parameters)
     
    # get line colors for plots 
    line_colors   = cm.inferno(np.linspace(0,0.9,len(results.segments)))       
  
    fig = plt.figure(save_filename)
    fig.set_size_inches(width,height)

    prev_seg_fuel       = 0
    prev_seg_extra_fuel = 0
    total_fuel          = 0

    axes = plt.subplot(1,1,1)
            
    for i in range(len(results.segments)):

        segment  = results.segments[i]
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min  
        segment_tag  =  results.segments[i].tag
        segment_name = segment_tag.replace('_', ' ')        

        if "has_additional_fuel" in segment.conditions.weights and segment.conditions.weights.has_additional_fuel == True:


            fuel     = segment.conditions.weights.fuel_mass[:,0]
            alt_fuel = segment.conditions.weights.additional_fuel_mass[:,0]

            if i == 0:

                plot_fuel     = np.negative(fuel)
                plot_alt_fuel = np.negative(alt_fuel)

                axes.plot( time , plot_fuel , 'ro-', marker = ps.marker, linewidth = ps.line_width , label = 'fuel')
                axes.plot( time , plot_alt_fuel , 'bo-', marker = ps.marker, linewidth = ps.line_width, label = 'additional fuel' )
                axes.plot( time , np.add(plot_fuel, plot_alt_fuel), 'go-', marker = ps.marker, linewidth = ps.line_width, label = 'total fuel' )

                axes.legend(loc='center right')   

            else:
                prev_seg_fuel       += results.segments[i-1].conditions.weights.fuel_mass[-1]
                prev_seg_extra_fuel += results.segments[i-1].conditions.weights.additional_fuel_mass[-1]

                current_fuel         = np.add(fuel, prev_seg_fuel)
                current_alt_fuel     = np.add(alt_fuel, prev_seg_extra_fuel)

                axes.plot( time , np.negative(current_fuel)  , 'ro-' , marker = ps.marker, linewidth = ps.line_width)
                axes.plot( time , np.negative(current_alt_fuel ), 'bo-', marker = ps.marker, linewidth = ps.line_width)
                axes.plot( time , np.negative(current_fuel + current_alt_fuel), 'go-', marker = ps.marker, linewidth = ps.line_width)

        else:
            
            initial_weight  = results.segments[0].conditions.weights.total_mass[:,0][0] 
            fuel            = results.segments[i].conditions.weights.total_mass[:,0]
            time            = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min 
            total_fuel      = np.negative(results.segments[i].conditions.weights.total_mass[:,0] - initial_weight )
            axes.plot( time, total_fuel, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)

    axes.set_ylabel('Fuel (kg)')
    axes.set_xlabel('Time (min)')

    set_axes(axes)  

    if show_legend:    
        leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
        leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
    
    # Adjusting the sub-plots for legend 
    fig.subplots_adjust(top=0.75)
    
    # set title of plot 
    title_text    = 'Aircraft Fuel Burnt'      
    fig.suptitle(title_text) 

    if save_figure:
        plt.savefig(save_filename + file_type)  
        
    return
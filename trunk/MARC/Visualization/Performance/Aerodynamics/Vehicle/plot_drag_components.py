## @ingroup Visualization-Performance-Aerodynamics
# plot_drag_components.py
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
#   Drag Components
# ---------------------------------------------------------------------- 

## @ingroup Visualization-Performance-Aerodynamics
def plot_drag_components(results,
                         save_figure=False,
                         show_legend= True,
                         save_filename="Drag_Components",
                         file_type=".png",
                        width = 12, height = 7):
    """This plots the drag components of the aircraft
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.condtions.aerodynamics.drag_breakdown
          parasite.total
          induced.total
          compressible.total
          miscellaneous.total
          
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
    fig.set_size_inches(12,height)
    
    for i in range(len(results.segments)): 
        time   = results.segments[i].conditions.frames.inertial.time[:,0] / Units.min 
        drag_breakdown = results.segments[i].conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cde = np.ones_like(cdm)*drag_breakdown.drag_coefficient_increment
        cd  = drag_breakdown.total[:,0]
         
            
        segment_tag  =  results.segments[i].tag
        segment_name = segment_tag.replace('_', ' ')
        
        
        axes_1 = plt.subplot(3,2,1)
        axes_1.plot(time, cdp, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)
        axes_1.set_ylabel(r'$C_{Dp}$')
        set_axes(axes_1)    
        
        axes_2 = plt.subplot(3,2,2)
        axes_2.plot(time,cdi, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width) 
        axes_2.set_ylabel(r'$C_{Di}$')
        set_axes(axes_2) 

        axes_3 = plt.subplot(3,2,3)
        axes_3.plot(time, cdc, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width) 
        axes_3.set_ylabel(r'$C_{Dc}$')
        set_axes(axes_3) 
        
        axes_4 = plt.subplot(3,2,4)
        axes_4.plot(time, cdm, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_4.set_ylabel(r'$C_{Dm}$')
        set_axes(axes_4)    
        
        axes_5 = plt.subplot(3,2,5)
        axes_5.plot(time, cde, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_5.set_xlabel('Time (mins)')
        axes_5.set_ylabel(r'$C_{De}$')
        set_axes(axes_5) 

        axes_6 = plt.subplot(3,2,6)
        axes_6.plot(time, cd, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_6.set_xlabel('Time (mins)')
        axes_6.set_ylabel(r'$C_D$')
        set_axes(axes_6) 
        
    
    if show_legend:                    
        leg =  fig.legend(bbox_to_anchor=(1.0, 0.5), loc='center right', ncol = 1) 
        leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
    
    # Adjusting the sub-plots for legend 
    fig.subplots_adjust(right=0.85)
    
    # set title of plot 
    title_text    = 'Drag Components'      
    fig.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return
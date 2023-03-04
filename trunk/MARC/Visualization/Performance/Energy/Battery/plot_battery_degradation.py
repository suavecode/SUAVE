## @ingroup Visualization-Performance-Energy-Battery
# plot_battery_degradation.py 
# 
# Created:    Jan 2023, M. Clarke
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
def plot_battery_degradation(results,
                            save_figure = False,
                            line_color = 'bo-',
                            line_color2 = 'rs--',
                            save_filename = "Battery_Degradation",
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

    fig  = plt.figure(save_filename)
    fig.set_size_inches(width,height) 
    
    num_segs          = len(results.segments)
    time_hrs          = np.zeros(num_segs)  
    capacity_fade     = np.zeros_like(time_hrs)
    resistance_growth = np.zeros_like(time_hrs)
    cycle_day         = np.zeros_like(time_hrs)
    charge_throughput = np.zeros_like(time_hrs)   
    
    for i in range(num_segs):   
        time_hrs[i]           = results.segments[i].conditions.frames.inertial.time[-1,0] / Units.hour
        cycle_day[i]          = results.segments[i].conditions.propulsion.battery.cell.cycle_in_day
        capacity_fade[i]      = results.segments[i].conditions.propulsion.battery.cell.capacity_fade_factor
        resistance_growth[i]  = results.segments[i].conditions.propulsion.battery.cell.resistance_growth_factor
        charge_throughput[i]  = results.segments[i].conditions.propulsion.battery.cell.charge_throughput[-1,0] 
        
 
    axes_1 = plt.subplot(3,2,1)
    axes_1.plot(charge_throughput, capacity_fade, color = ps.color , marker = ps.marker, linewidth = ps.line_width ) 
    axes_1.set_ylabel('$E/E_0$')
    axes_1.set_xlabel('Time (hrs)')
    set_axes(axes_1)      

    axes_2 = plt.subplot(3,2,3)
    axes_2.plot(time_hrs, capacity_fade, color = ps.color, marker = ps.marker, linewidth = ps.line_width ) 
    axes_2.set_ylabel('$E/E_0$')
    axes_2.set_xlabel('Time (hrs)')
    set_axes(axes_2)     

    axes_3 = plt.subplot(3,2,5)
    axes_3.plot(cycle_day, capacity_fade, color = ps.color, marker = ps.marker, linewidth = ps.line_width ) 
    axes_3.set_ylabel('$E/E_0$')
    axes_3.set_xlabel('Time (days)')
    set_axes(axes_3)     

    axes_4 = plt.subplot(3,2,2) 
    axes_4.plot(charge_throughput, resistance_growth, color = ps.color, marker = ps.marker, linewidth = ps.line_width )
    axes_4.set_ylabel('$R/R_0$')
    axes_4.set_xlabel('Time (hrs)')
    set_axes(axes_4)      

    axes_5 = plt.subplot(3,2,4) 
    axes_5.plot(time_hrs, resistance_growth, color = ps.color, marker = ps.marker, linewidth = ps.line_width )
    axes_5.set_ylabel('$R/R_0$')
    axes_5.set_xlabel('Time (hrs)')
    set_axes(axes_5)     

    axes_6 = plt.subplot(3,2,6) 
    axes_6.plot(cycle_day, resistance_growth, color = ps.color, marker = ps.marker, linewidth = ps.line_width )
    axes_6.set_ylabel('$R/R_0$')
    axes_6.set_xlabel('Time (days)')
    set_axes(axes_6)             
        
    # set title of plot 
    title_text    = 'Battery Cell Degradation'      
    fig.suptitle(title_text) 
    
    plt.tight_layout()    
    if save_figure:    
        fig.savefig(save_filename + file_type) 
    
    return


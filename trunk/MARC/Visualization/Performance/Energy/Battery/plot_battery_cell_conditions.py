## @ingroup Visualization-Performance-Energy-Battery
# plot_battery_cell_conditions.py
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
def plot_battery_cell_conditions(results,
                                  save_figure = False,
                                  show_legend = True,
                                  save_filename = "Battery_Cell_Conditions",
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
        cell_power          = results.segments[i].conditions.propulsion.battery.cell.power[:,0]
        cell_energy         = results.segments[i].conditions.propulsion.battery.cell.energy[:,0]
        cell_volts          = results.segments[i].conditions.propulsion.battery.cell.voltage_under_load[:,0]
        cell_volts_oc       = results.segments[i].conditions.propulsion.battery.cell.voltage_open_circuit[:,0]
        cell_current        = results.segments[i].conditions.propulsion.battery.cell.current[:,0]
        cell_SOC            = results.segments[i].conditions.propulsion.battery.cell.state_of_charge[:,0]
        cell_temp           = results.segments[i].conditions.propulsion.battery.cell.temperature[:,0]
        cell_charge         = results.segments[i].conditions.propulsion.battery.cell.charge_throughput[:,0]
    
        cell_battery_amp_hr = (cell_energy/ Units.Wh )/cell_volts
        cell_C_instant      = cell_current/cell_battery_amp_hr
        cell_C_nominal      = cell_current/np.max(cell_battery_amp_hr)

        segment_tag  =  results.segments[i].tag
        segment_name = segment_tag.replace('_', ' ') 
        
        axes_1 = plt.subplot(4,2,1)
        axes_1.plot(time, cell_SOC, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width, label = segment_name)
        axes_1.set_ylabel(r'SOC')
        set_axes(axes_1)     
        
        axes_2 = plt.subplot(4,2,2)
        axes_2.plot(time, cell_energy/Units.Wh, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_2.set_ylabel(r'Energy (W-hr)')
        set_axes(axes_2) 

        axes_3 = plt.subplot(4,2,3)
        axes_3.plot(time, cell_current, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_3.set_ylabel(r'Current (A)')
        set_axes(axes_3)  

        axes_4 = plt.subplot(4,2,4)
        axes_4.plot(time, -cell_power, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_4.set_ylabel(r'Power (W)')
        set_axes(axes_4)     
        
        axes_5 = plt.subplot(4,2,5)
        axes_5.plot(time, cell_volts, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_5.set_ylabel(r'Voltage (V)')
        set_axes(axes_5) 

        axes_6 = plt.subplot(4,2,6)
        axes_6.plot(time, cell_volts_oc, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_6.set_ylabel(r'Voltage OC (V)')
        set_axes(axes_6)  

        axes_7 = plt.subplot(4,2,7)
        axes_7.plot(time, cell_C_instant, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_7.set_ylabel(r'Inst. C-Rate (C)')
        axes_7.set_xlabel('Time (mins)')
        set_axes(axes_7)     
        
        axes_8 = plt.subplot(4,2,8)
        axes_8.plot(time, cell_C_nominal, color = line_colors[i], marker = ps.marker, linewidth = ps.line_width)
        axes_8.set_ylabel(r'Nom. C-Rate (C)')
        axes_8.set_xlabel('Time (mins)')
        set_axes(axes_8) 
 
    
    if show_legend:        
        leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
        leg.set_title('Flight Segment', prop={'size': ps.legend_font_size, 'weight': 'heavy'})    
    
    # Adjusting the sub-plots for legend 
    fig.subplots_adjust(top=0.8)
    
    # set title of plot 
    title_text    = 'Battery Cell Conditions'      
    fig.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return
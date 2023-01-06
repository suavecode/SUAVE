## @ingroup Visualization-Performance-Energy-Battery
# plot_battery_degradation.py 
# 
# Created:    Jan 2023, M. Clarke
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Common import plot_style, save_plot
from SUAVE.Visualization.Performance.Energy.Battery import *
from SUAVE.Visualization.Performance.Energy.Common import *

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Visualization-Performance-Energy-Battery
def plot_battery_degradation(results,
                                  save_figure = False,
                                  save_filename = "Battery_Degradation",
                                  file_type = ".png",
                                  width = 1200, height = 600,
                                  *args, **kwargs):
    """This plots the solar flux and power train performance of an solar powered aircraft

    Assumptions:
    None
    
    Deprecated SUAVE Mission Plots Functions 
    
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
    
    # Create empty dataframe to be populated by the segment data  
    num_segs          = len(results.segments)
    time_hrs          = np.zeros(num_segs)
    capacity_fade     = np.zeros_like(time_hrs)
    resistance_growth = np.zeros_like(time_hrs)
    cycle_day         = np.zeros_like(time_hrs)
    charge_throughput = np.zeros_like(time_hrs)

    for i in range(num_segs):
        time_hrs[i]          = results.segments[i].conditions.frames.inertial.time[-1,0] / Units.hour
        cycle_day[i]         = results.segments[i].conditions.propulsion.battery.cell.cycle_in_day
        capacity_fade[i]     = results.segments[i].conditions.propulsion.battery.cell.capacity_fade_factor
        resistance_growth[i] = results.segments[i].conditions.propulsion.battery.cell.resistance_growth_factor
        charge_throughput[i] = results.segments[i].conditions.propulsion.battery.cell.charge_throughput[-1,0] 
        

    # Set plot properties
    fig = make_subplots(rows=2, cols=3) 
    fig.add_trace(go.Scatter(
        x=charge_throughput,
        y=capacity_fade,
        showlegend=False),
        row=1, col=1)

    fig.add_trace(go.Scatter(
        x=charge_throughput,
        y= resistance_growth,
        showlegend=False),
        row=2, col=1) 
    
    fig.add_trace(go.Scatter(
        x=time_hrs,
        y=capacity_fade, 
        showlegend=False),
        row=1, col=2)   
    
    fig.add_trace(go.Scatter(
        x=time_hrs,
        y=resistance_growth, 
        showlegend=False),
        row=2, col=2) 
    
    fig.add_trace(go.Scatter(
        x=cycle_day,
        y=capacity_fade, 
        showlegend=False),
        row=1, col=3)    
    
    fig.add_trace(go.Scatter(
        x=cycle_day,
        y=resistance_growth, 
        showlegend=False),
        row=2, col=3)          


    fig.update_yaxes(title_text='% Capacity Fade', row=1, col=1)
    fig.update_yaxes(title_text='% Resistance Growth', row=2, col=1) 
    fig.update_yaxes(title_text='% Capacity Fade', row=1, col=2) 
    fig.update_yaxes(title_text='% Resistance Growth', row=2, col=2)
    fig.update_yaxes(title_text='% Capacity Fade', row=1, col=3) 
    fig.update_yaxes(title_text='% Resistance Growth', row=2, col=3) 
    fig.update_xaxes(title_text='Chargethrough (Ah)', row=2, col=1)
    fig.update_xaxes(title_text='Time (hr)', row=2, col=2)
    fig.update_xaxes(title_text='Day', row=2, col=2)

    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Battery Degradation'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

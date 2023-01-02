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

    plot_cols = [
        "Cycle",
        "C", 
        "R", 
        "Charge",
        "Segment"
    ]

    df = pd.DataFrame(columns=plot_cols)

    # Get the segment-by-segment results


    num_segs          = len(results.segments)
    time_hrs          = np.zeros(num_segs)
    capacity_fade     = np.zeros_like(time_hrs)
    resistance_growth = np.zeros_like(time_hrs)
    cycle_day         = np.zeros_like(time_hrs)
    charge_throughput = np.zeros_like(time_hrs)

    for segment in results.segments.values():
        time_hrs            = segment.conditions.frames.inertial.time[-1,0] / Units.hour
        cycle_day           = segment.conditions.propulsion.battery_cycle_day
        capacity_fade       = segment.conditions.propulsion.battery_capacity_fade_factor
        resistance_growth   = segment.conditions.propulsion.battery_resistance_growth_factor
        charge_throughput   = segment.conditions.propulsion.battery_cell_charge_throughput[-1,0] 
        
        segment_frame = pd.DataFrame(
            np.column_stack((
                cycle_day,
                capacity_fade,
                resistance_growth,
                charge_throughput,

            )),
            columns=plot_cols[:-1], index=time_hrs
        )

        segment_frame['Segment'] = [segment.tag for i in range(len(time_hrs))]

        # Append to collecting dataframe

        df = df.append(segment_frame)

    # Set plot properties

    fig = make_subplots(rows=2, cols=3)

    # Add traces to the figure for each value by segment

    for seg, data in df.groupby("Segment", sort=False):

        seg_name = ' '.join(seg.split("_")).capitalize()

        fig.add_trace(go.Scatter(
            x=data['Charge'],
            y=data['C'],
            name=seg_name),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data['Charge'],
            y= data['R'],
            name=seg_name),
            row=2, col=1)
        
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y= data['C'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)   
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['R'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)
        
        
        fig.add_trace(go.Scatter(
            x=data['Cycle'],
            y= data['C'],
            name=seg_name,
            showlegend=False),
            row=1, col=3)    
        
        fig.add_trace(go.Scatter(
            x=data['Cycle'],
            y=data['R'],
            name=seg_name,
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

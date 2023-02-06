## @ingroup Visualization-Performance-Energy-Battery
# plot_solar_flux.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from MARC.Core import Units
from MARC.Visualization.Performance.Common import plot_style, save_plot
from MARC.Visualization.Performance.Energy.Battery import *
from MARC.Visualization.Performance.Energy.Common import *

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Visualization-Performance-Energy-Battery
def plot_solar_flux(results,
                                  save_figure = False,
                                  save_filename = "Solar_Flux",
                                  file_type = ".png",
                                  width = 1200, height = 600,
                                  *args, **kwargs):
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
    # Create empty dataframe to be populated by the segment data

    plot_cols = [
        "Solar Flux",
        "Charging Power",
        "Battery Current",
        "Battery Energy",
        "Segment"
    ]

    df = pd.DataFrame(columns=plot_cols)

    # Get the segment-by-segment results

    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        flux   = segment.conditions.propulsion.solar_flux[:,0]
        charge = segment.conditions.propulsion.battery.pack.power_draw[:,0]
        current= segment.conditions.propulsion.battery.pack.current[:,0]
        energy = segment.conditions.propulsion.battery.pack.energy[:,0] / Units.MJ

        segment_frame = pd.DataFrame(
            np.column_stack((
                flux,
                charge,
                current,
                energy,

            )),
            columns=plot_cols[:-1], index=time
        )

        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting dataframe

        df = df.append(segment_frame)

    # Set plot properties

    fig = make_subplots(rows=2, cols=2)

    # Add traces to the figure for each value by segment

    for seg, data in df.groupby("Segment", sort=False):

        seg_name = ' '.join(seg.split("_")).capitalize()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Solar Flux'],
            name=seg_name),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Charging Power'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Battery Current'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Battery Energy'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)

    fig.update_yaxes(title_text='Solar Flux (W/m^2)', row=1, col=1)
    fig.update_yaxes(title_text='Charging Power (W)', row=1, col=2)
    fig.update_yaxes(title_text='Battery Current (A)', row=2, col=1) 
    fig.update_yaxes(title_text='Battery Energy (MJ)', row=2, col=2) 

    fig.update_xaxes(title_text='Time (min)', row=2, col=1)
    fig.update_xaxes(title_text='Time (min)', row=2, col=2)

    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Solar Flux Conditions'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return
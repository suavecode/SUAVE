## @ingroup Visualization-Performance-Energy-Fuel
# plot_altitude_sfc_weight.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Common import plot_style, save_plot

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Visualization-Performance-Energy-Fuel
def plot_altitude_sfc_weight(results,
                             save_figure = False,
                             save_filename = "Altitude_SFC_Weight" ,
                             file_type = ".png",
                             width=800,height=500,
                             *args, **kwargs):
    """This plots the altitude, specific fuel consumption and vehicle weight.

    Assumptions:
    None

    Source:

    Depricated SUAVE Mission Plots Functions

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

    # Create empty data frame to be populated by the segment data

    df = pd.DataFrame(columns=['Altitude', 'SFC', 'Mass', 'Segment'])

    # Get the segment-by-segment results for altitude, mass, and the
    # SFC (calculated)

    for segment in results.segments.values():
        time = segment.conditions.frames.inertial.time[:, 0] / Units.min
        mass = segment.conditions.weights.total_mass[:, 0] / Units.lb
        altitude = segment.conditions.freestream.altitude[:, 0] / Units.ft
        mdot = segment.conditions.weights.vehicle_mass_rate[:, 0]
        thrust = segment.conditions.frames.body.thrust_force_vector[:, 0]
        sfc = (mdot / Units.lb) / (thrust / Units.lbf) * Units.hr

        # Assemble data into temporary holding data frame

        segment_frame = pd.DataFrame(
            np.column_stack((altitude, sfc, mass)),
            columns=['Altitude', 'SFC', 'Mass'], index=time)
        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting data-frame

        df = df.append(segment_frame)

    # Set plot parameters

    fig = make_subplots(rows=3, cols=1,
                        vertical_spacing=0.05)

    # Add traces to the figure for each value by segment.

    for seg, data in df.groupby("Segment"):
        seg_name = ' '.join(seg.split("_")).capitalize()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Altitude'],
            name=seg_name),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Mass'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SFC'],
            name=seg_name,
            showlegend=False),
            row=3, col=1)

    # Set sublot axis titles

    fig.update_yaxes(title_text='Altitude (ft)', row=1, col=1)
    fig.update_yaxes(title_text='Weight (lb)', row=2, col=1)
    fig.update_yaxes(title_text='SFC (lb/lbf-hr)', row=3, col=1)

    fig.update_xaxes(title_text='Time (min)', row=3, col=1)

    # Set overall figure layout style and legend title

    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Altitude SFC and Weight'
        
    )

    # Update Figure Style and Show 
    fig = plot_style(fig)
    fig.show()

    # Optionally save the figure with kaleido import check

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

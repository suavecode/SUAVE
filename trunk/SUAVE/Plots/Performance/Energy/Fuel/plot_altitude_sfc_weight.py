## @ingroup Plots-Performance-Energy-Fuel
# plot_altitude_sfc_weight.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from itertools import cycle

from SUAVE.Core import Units

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Plots-Performance-Energy-Fuel
def plot_altitude_sfc_weight(results,
                             segment_colors = 'Dark24',
                             save_figure = False,
                             save_filename = "Altitude_SFC_Weight" ,
                             file_type = ".png",
                             width=800,
                             height=500):
    """This plots the altitude, specific fuel consumption and vehicle weight

    Assumptions:
    None

    Source:
    None

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

    # Set plot parameters (3 x 1 subplots with grid lines showing)

    fig = make_subplots(rows=3, cols=1,
                        vertical_spacing=0.05)

    # Create cyclic iterator for segment-by-segment coloration

    coloriter = cycle(plotly.colors.get_colorscale(segment_colors))

    # Add traces to the figure for each value by segment. Note that technically
    # only the altitude trace is shown in the legend, but colors are matched for
    # each value, so the legend remains accurate.

    for seg, data in df.groupby("Segment"):
        seg_name = ' '.join(seg.split("_")).capitalize()
        seg_color = next(coloriter)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Altitude'],
            marker=dict(size=3, color=seg_color),
            name=seg_name,
            line=dict(color=seg_color)),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Mass'],
            marker=dict(size=3, color=seg_color),
            name=seg_name,
            line=dict(color=seg_color),
            showlegend=False),
            row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SFC'],
            marker=dict(size=3, color=seg_color),
            name=seg_name,
            line=dict(color=seg_color),
            showlegend=False),
            row=3, col=1)

    # Set axis titles and gridlines using a dictionary of keyword arguments

    plot_kwargs = dict(
        ticks='outside', tickwidth=2, ticklen=6,
        showline=True, linewidth=2, linecolor='black',
        showgrid=True, gridwidth=1, gridcolor='grey',
        zeroline=True, zerolinewidth=1, zerolinecolor='black'
    )

    fig.update_yaxes(title_text='Altitude (ft)',
                     **plot_kwargs,
                     row=1, col=1)
    fig.update_yaxes(title_text='Weight (lb)',
                     **plot_kwargs,
                     row=2, col=1)
    fig.update_yaxes(title_text='SFC (lb/lbf-hr)',
                     **plot_kwargs,
                     row=3, col=1)

    fig.update_xaxes(title_text='Time (min)',
                     **plot_kwargs,
                     row=3, col=1)
    fig.update_xaxes(**plot_kwargs,
                     row=2, col=1)
    fig.update_xaxes(**plot_kwargs,
                     row=1, col=1)

    # Set overall figure layout style and legend title

    fig.update_layout(
        plot_bgcolor='white',
        width=width, height=height,
        margin=dict(t=0, l=0, b=0, r=0),
        legend_title_text='Segment',
    )

    # Show the figure
    fig.show()

    if save_figure:
        try:
            import kaleido
            fig.write_image(save_filename.replace("_", " ") + file_type)
        except ImportError:
            raise ImportError(
                'You need to install kaleido to save the figure.')


    return

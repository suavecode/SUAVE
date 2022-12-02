## @ingroup Plots-Performance-Energy-Common
# plot_disc_power_loading.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units
from SUAVE.Plots.Performance.Common import plot_style, save_plot

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Plots-Performance-Energy-Common
def plot_disc_power_loading(results,
                            save_figure=False,
                            save_filename="Disc_Power_Loading",
                            file_type = ".png",
                            width = 1600, height = 800,
                            *args, **kwargs):
    """Plots propeller disc and power loadings

    Assumptions:
    None

    Source:

    Deprecated SUAVE Mission Plots Functions

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge

    results.segments.conditions.propulsion.
        disc_loadings
        power_loading


    Outputs:
    Plots

    Properties Used:
    N/A
    """

    # Create empty dataframe to be populated by the segment data

    plot_cols = [
        'Disc Power',
        'Power Loading',
        'Segment'
    ]

    for segment in results.segments.values():

        time  = segment.conditions.frames.inertial.time[:,0] / Units.min
        DL    = segment.conditions.propulsion.disc_loading
        PL    = segment.conditions.propulsion.power_loading

        # Assemble data into a temporary holding dataframe

        segment_frame = pd.DataFrame(
            np.column_stack((
                DL,
                PL
            )),
            columns=plot_cols[:-1], index=time
        )

        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting dataframe

        df = df.append(segment_frame)

    # Set plot parameters

    fig = make_subplots(rows=2, cols=2)

    # Add traces to the figure for each value by segment

    for seg, data in df.groupby("Segment", sort=False):
        seg_name = ' '.join(seg.split("_")).capitalize()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Disc Power'],
            name=seg_name),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Power Loading'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)

    fig.update_yaxes(title_text='Lift Disc Power N/m^2', row=1, col=1)
    fig.update_yaxes(title_text='Lift Power Loading (N/W)', row=2, col=1)

    fig.update_xaxes(title_text='Time (min)', row=2, col=1)

    # Set overall figure layout style and legend title

    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Blade Loadings'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

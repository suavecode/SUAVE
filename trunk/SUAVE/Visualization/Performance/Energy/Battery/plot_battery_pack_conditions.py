## @ingroup Visualization-Performance-Energy-Battery
# plot_battery_pack_conditions.py
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

## @ingroup Visualization-Performance-Energy-Battery
def plot_battery_pack_conditions(results,
                                 save_figure=False,
                                 save_filename="Battery_Pack_Conditions",
                                 file_type=".png",
                                 width = 1200, height = 600,
                                 *args, **kwargs):
    """Plots the pack-level conditions of the battery throughout flight.

    Assumptions:
    None

    Source:

    Deprecated SUAVE Mission Plots Functions

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

    # Create empty dataframe to be populated by the segment data
    plot_cols = ['SOC',
                 'Energy',
                 'Power',
                 'Current',
                 'Voltage',
                 'Voltage OC',
                 'C Rate Instant',
                 'C Rate Nominal',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)

    # Get the segment-by-segment results
    for segment in results.segments.values():

        time                = segment.conditions.frames.inertial.time[:,0] / Units.min
        pack_power          = segment.conditions.propulsion.battery.pack.power_draw[:,0]
        pack_energy         = segment.conditions.propulsion.battery.pack.energy[:,0]
        pack_volts          = segment.conditions.propulsion.battery.pack.voltage_under_load[:,0]
        pack_volts_oc       = segment.conditions.propulsion.battery.pack.voltage_open_circuit[:,0]
        pack_current        = segment.conditions.propulsion.battery.pack.current[:,0]
        pack_SOC            = segment.conditions.propulsion.battery.cell.state_of_charge[:,0]

        pack_battery_amp_hr = (pack_energy/ Units.Wh )/pack_volts
        pack_C_instant      = pack_current/pack_battery_amp_hr
        pack_C_nominal      = pack_current/np.max(pack_battery_amp_hr)

        # Assemble data into temporary holding data frame
        segment_frame = pd.DataFrame(
            np.column_stack((pack_SOC,
                             (pack_energy/Units.Wh),
                             -pack_power,
                             pack_current,
                             pack_volts,
                             pack_volts_oc,
                             pack_C_instant,
                             pack_C_nominal)
                             ),
            columns = plot_cols[:-1], index=time
        )
        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting data frame
        df = df.append(segment_frame)

    # Set plot parameters
    fig = make_subplots(rows=4, cols=2)

    # Add traces to the figure for each value by segment
    for seg, data in df.groupby("Segment", sort=False):
        seg_name = ' '.join(seg.split("_")).capitalize()

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SOC'],
            name=seg_name),
            row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Energy'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Power'],
            name=seg_name,
            showlegend=False),
            row=3, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Current'],
            name=seg_name,
            showlegend=False),
            row=4, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Voltage'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Voltage OC'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['C Rate Instant'],
            name=seg_name,
            showlegend=False),
            row=3, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['C Rate Nominal'],
            name=seg_name,
            showlegend=False),
            row=4, col=2)

    # Add subplot axis titles
    fig.update_yaxes(title_text='SOC', row=1, col=1)
    fig.update_yaxes(title_text='Energy (W-hr)', row=2, col=1)
    fig.update_yaxes(title_text='Power (W)', row=3, col=1)
    fig.update_yaxes(title_text='Current (A)', row=4, col=1)
    fig.update_yaxes(title_text='Voltage (V)', row=1, col=2)
    fig.update_yaxes(title_text='Voltage OC (V)', row=2, col=2)
    fig.update_yaxes(title_text='Inst. C-Rate (C)', row=3, col=2)
    fig.update_yaxes(title_text='Nom. C-Rate (C)', row=4, col=2)

    fig.update_xaxes(title_text='Time (min)', row=4, col=1)
    fig.update_xaxes(title_text='Time (min)', row=4, col=2)

    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Battery Pack Conditions'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

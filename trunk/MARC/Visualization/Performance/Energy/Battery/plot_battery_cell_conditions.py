## @ingroup Visualization-Performance-Energy-Battery
# plot_battery_cell_conditions.py
# 
# Created:    Nov 2022, J. Smart
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Units
from MARC.Visualization.Performance.Common import plot_style, save_plot

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Visualization-Performance-Energy-Battery
def plot_battery_cell_conditions(results,
                                 save_figure=False,
                                 show_figure = True,
                                 save_filename="Battery_Cell_Conditions",
                                 file_type = ".png",
                                 width = 1200, height = 600,
                                 *args, **kwargs):
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

    # Create empty dataframe to be populated by the segment data

    plot_cols = ['SOC',
                 'Energy',
                 'Power',
                 'Voltage',
                 'Voltage OC',
                 'C Rate Instant',
                 'C Rate Nominal',
                 'Current',
                 'Charge Throughput',
                 'Temperature',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)

    # Get the segment-by-segment results

    for segment in results.segments.values():

        time                = segment.conditions.frames.inertial.time[:,0] / Units.min
        cell_power          = segment.conditions.propulsion.battery.cell.power[:,0]
        cell_energy         = segment.conditions.propulsion.battery.cell.energy[:,0]
        cell_volts          = segment.conditions.propulsion.battery.cell.voltage_under_load[:,0]
        cell_volts_oc       = segment.conditions.propulsion.battery.cell.voltage_open_circuit[:,0]
        cell_current        = segment.conditions.propulsion.battery.cell.current[:,0]
        cell_SOC            = segment.conditions.propulsion.battery.cell.state_of_charge[:,0]
        cell_temp           = segment.conditions.propulsion.battery.cell.temperature[:,0]
        cell_charge         = segment.conditions.propulsion.battery.cell.charge_throughput[:,0]

        cell_battery_amp_hr = (cell_energy/ Units.Wh )/cell_volts
        cell_C_instant      = cell_current/cell_battery_amp_hr
        cell_C_nominal      = cell_current/np.max(cell_battery_amp_hr)

        # Assemble data into temporary holding data frame

        segment_frame = pd.DataFrame(
            np.column_stack((cell_SOC,
                             (cell_energy/Units.Wh),
                             -cell_power,
                             cell_volts,
                             cell_volts_oc,
                             cell_C_instant,
                             cell_C_nominal,
                             cell_current,
                             cell_charge,
                             cell_temp)),
            columns = plot_cols[:-1], index=time
        )
        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting data frame

        df = df.append(segment_frame)

    # Set plot parameters

    fig = make_subplots(rows=5, cols=2)

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
            y=data['Voltage'],
            name=seg_name,
            showlegend=False),
            row=4, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Voltage OC'],
            name=seg_name,
            showlegend=False),
            row=5, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['C Rate Instant'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['C Rate Nominal'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Current'],
            name=seg_name,
            showlegend=False),
            row=3, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Charge Throughput'],
            name=seg_name,
            showlegend=False),
            row=4, col=2)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Temperature'],
            name=seg_name,
            showlegend=False),
            row=5, col=2)

    # Add subplot axis titles

    fig.update_yaxes(title_text='SOC', row=1, col=1)
    fig.update_yaxes(title_text='Energy (W-hr)', row=2, col=1)
    fig.update_yaxes(title_text='Power (W)', row=3, col=1)
    fig.update_yaxes(title_text='Voltage (V)', row=4, col=1)
    fig.update_yaxes(title_text='Voltage OC (V)', row=5, col=1)
    fig.update_yaxes(title_text='Inst. C-Rate (C)', row=1, col=2)
    fig.update_yaxes(title_text='Nom. C-Rate (C)', row=2, col=2)
    fig.update_yaxes(title_text='Current (A)', row=3, col=2)
    fig.update_yaxes(title_text='Charge Throughput (Ah)', row=4, col=2)
    fig.update_yaxes(title_text='Temperature (K)', row=5, col=2)

    fig.update_xaxes(title_text='Time (min)', row=5, col=1)
    fig.update_xaxes(title_text='Time (min)', row=5, col=2)

    # Set overall figure layout style and legend title

    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Battery Cell Conditions'
    )

    fig = plot_style(fig)
    if show_figure:
        fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

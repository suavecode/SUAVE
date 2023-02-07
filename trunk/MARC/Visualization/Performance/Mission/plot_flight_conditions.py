## @defgroup Visualization-Performance-Mission 
# plot_flight_conditions.py
# 
# Created:    Dec 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Units
from MARC.Visualization.Performance.Common import plot_style, save_plot

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------- 
#   Flight Condition
# ---------------------------------------------------------------------- 
## @defgroup Visualization-Performance-Mission 
def plot_flight_conditions(results,
                            save_figure=False,
                            save_filename="Flight Conditions",
                            file_type=".png",
                            show_figure = True,
                            width = 1200, height = 600,
                            *args, **kwargs):
    """This plots the flights the conditions

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
         frames
             body.inertial_rotations
             inertial.position_vector
         freestream.velocity
         aerodynamics.
             lift_coefficient
             drag_coefficient
             angle_of_attack

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    
    plot_cols = ['altitude',
                 'airspeed',
                 'theta',
                 'Range',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)    

    

    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        airspeed = segment.conditions.freestream.velocity[:,0] /   Units['mph']
        theta    = segment.conditions.frames.body.inertial_rotations[:,1,None] / Units.deg
        Range    = segment.conditions.frames.inertial.aircraft_range[:,0]/ Units.nmi
        altitude = segment.conditions.freestream.altitude[:,0]/Units.feet
        
        segment_frame = pd.DataFrame(
        np.column_stack((altitude,
                         airspeed,
                         theta,
                         Range,
                         )),
        columns = plot_cols[:-1], index=time
        )

        segment_frame['Segment'] = [segment.tag for i in range(len(time))]
    
        # Append to collecting data frame
        df = df.append(segment_frame)               
    
    # Set plot parameters
    fig = make_subplots(rows=2, cols=2)

    # Add traces to the figure for each value by segment
    for seg, data in df.groupby("Segment", sort=False):
        seg_name = ' '.join(seg.split("_")).capitalize()  
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['altitude'],
            name=seg_name),
            row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['airspeed'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['theta'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Range'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)        
    
    fig.update_yaxes(title_text='Altitude (ft)', row=1, col=1)
    fig.update_yaxes(title_text='Airspeed (mph)', row=1, col=2)
    fig.update_yaxes(title_text='Pitch Angle (deg)', row=2, col=1)
    fig.update_yaxes(title_text='Range (nmi)', row=2, col=2)
    
    fig.update_xaxes(title_text='Time (min)', row=2, col=1)
    fig.update_xaxes(title_text='Time (min)', row=2, col=2)    

    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Flight Conditions'
    )

    fig = plot_style(fig)
    if show_figure:
        fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

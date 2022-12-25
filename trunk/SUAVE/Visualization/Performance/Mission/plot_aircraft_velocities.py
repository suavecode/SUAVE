## @defgroup Visualization-Performance-Mission 
# plot_aircraft_velocities.py
# 
# Created:    Dec 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Common import plot_style, save_plot

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------- 
#   Aircraft Velocities
# ---------------------------------------------------------------------- 
## @defgroup Visualization-Performance-Mission 
def plot_aircraft_velocities(results,
                            save_figure=False,
                            save_filename="Aircraft Velocities",
                            file_type=".png",
                            width = 1400, height = 665,
                            *args, **kwargs):
    """This plots true, equivalent, and calibrated airspeeds along with mach

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.condtions.freestream.
        velocity
        density
        mach_number

    Outputs:
    Plots

    Properties Used:
    N/A
    """
    
    plot_cols = ['TAS',
                 'EAS',
                 'CAS',
                 'mach',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)    
    
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        velocity = segment.conditions.freestream.velocity[:,0] / Units.kts
        density  = segment.conditions.freestream.density[:,0]
        PR       = density/1.225
        EAS      = velocity * np.sqrt(PR)
        mach     = segment.conditions.freestream.mach_number[:,0]
        CAS      = EAS * (1+((1/8)*((1-PR)*mach**2))+((3/640)*(1-10*PR+(9*PR**2)*(mach**4))))

        
        segment_frame = pd.DataFrame(
        np.column_stack((velocity,
                         EAS,
                         CAS,
                         mach)),
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
            y=data['TAS'],
            name=seg_name),
            row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['EAS'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['CAS'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['mach'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)        
    
    fig.update_yaxes(title_text='True Airspeed (kts)', row=1, col=1)
    fig.update_yaxes(title_text='Equivalent Airspeed (kts)', row=1, col=2)
    fig.update_yaxes(title_text='Calibrated Airspeed (kts)', row=2, col=1)
    fig.update_yaxes(title_text='Mach Number', row=2, col=2)
    
    fig.update_xaxes(title_text='Time (min)', row=2, col=1)
    fig.update_xaxes(title_text='Time (min)', row=2, col=2)    

    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Flight Conditions'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return    


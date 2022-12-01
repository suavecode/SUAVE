## @ingroup Plots-Performance-Aerodynamics
# plot_stability_coefficients.py
# 
# Created:    Nov 2022, E. Botero
# Modified:   

# ----------------------------------------------------------------------
#   Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units
from SUAVE.Plots.Performance.Common import plot_style, save_plot

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------- 
#   Stability Coefficients
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_stability_coefficients(results,
                                save_figure=False,
                                save_filename="Stability_Coefficents",
                                file_type=".png",
                                width = 1600, height = 665,
                                *args, **kwargs):
    """This plots the static stability characteristics of an aircraft
    
    Assumptions:
    None
    
    Source:

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge
    
    
    Inputs:
    results.segments.conditions.stability.
       static
           CM
           Cm_alpha
           static_margin
       aerodynamics.
           angle_of_attack
    Outputs:
    
    Plots
    Properties Used:
    N/A
    """
    
    plot_cols = ['CM',
                 'CM_a',
                 'SM',
                 'AoA',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)
    

    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        cm       = segment.conditions.stability.static.CM[:,0]
        cm_alpha = segment.conditions.stability.static.Cm_alpha[:,0]
        SM       = segment.conditions.stability.static.static_margin[:,0]
        aoa      = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        
        segment_frame = pd.DataFrame(
        np.column_stack((cm,
                         cm_alpha,
                         SM,
                         aoa)),
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
            y=data['AoA'],
            name=seg_name),
            row=1, col=1)    
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['CM'],
            name=seg_name),
            row=1, col=2)        

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['CM_a'],
            name=seg_name),
            row=2, col=1)        

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SM'],
            name=seg_name),
            row=2, col=2)        

    fig.update_yaxes(title_text='AoA (deg)', row=1, col=1)
    fig.update_yaxes(title_text='$C_M$', row=1, col=2)
    fig.update_yaxes(title_text='$C_M\alpha$', row=2, col=1)
    fig.update_yaxes(title_text='Static Margin (%)', row=2, col=2)
    
    fig.update_xaxes(title_text='Time (min)', row=2, col=1)
    fig.update_xaxes(title_text='Time (min)', row=2, col=2)        

    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

## @ingroup Plots-Performance-Aerodynamics
# plot_aerodynamic_coefficients.py
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
#   Aerodynamic Coefficients
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_aerodynamic_coefficients(results,
                                  save_figure=False,
                                  save_filename="Aerodynamic_Coefficents",
                                  file_type=".png",
                                  width = 1600, height = 665,
                                  *args, **kwargs):
    """This plots the aerodynamic coefficients
    
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
    results.segments.condtions.aerodynamics.
        lift_coefficient
        drag_coefficient
        angle_of_attack
        
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """

    plot_cols = ['CL',
                 'CD',
                 'AoA',
                 'L_D',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)

    # Get the segment-by-segment results
    for segment in results.segments.values():
        time = segment.conditions.frames.inertial.time[:,0] / Units.min
        cl   = segment.conditions.aerodynamics.lift_coefficient[:,0,None]
        cd   = segment.conditions.aerodynamics.drag_coefficient[:,0,None]
        aoa  = segment.conditions.aerodynamics.angle_of_attack[:,0] / Units.deg
        l_d  = cl/cd    
        
        segment_frame = pd.DataFrame(
        np.column_stack((cl,
                         cd,
                         aoa,
                         l_d)),
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
            y=data['CL'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['CD'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['L_D'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)        
    
    fig.update_yaxes(title_text='AoA (deg)', row=1, col=1)
    fig.update_yaxes(title_text='CL', row=1, col=2)
    fig.update_yaxes(title_text='CD', row=2, col=2)
    fig.update_yaxes(title_text='L/D', row=2, col=1)
    
    fig.update_xaxes(title_text='Time (min)', row=2, col=1)
    fig.update_xaxes(title_text='Time (min)', row=2, col=2)    

    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Aerodynamic Coefficients'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)

    return

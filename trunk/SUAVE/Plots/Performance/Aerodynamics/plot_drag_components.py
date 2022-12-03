## @ingroup Plots-Performance-Aerodynamics
# plot_drag_components.py
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
#   Drag Components
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance-Aerodynamics
def plot_drag_components(results,
                         save_figure=False,
                         save_filename="Drag_Components",
                         file_type=".png",
                         width = 1600, height = 665,
                         *args, **kwargs):
    """This plots the drag components of the aircraft
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    results.segments.condtions.aerodynamics.drag_breakdown
          parasite.total
          induced.total
          compressible.total
          miscellaneous.total
          
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """
    
    # Create empty dataframe to be populated by the segment data
    plot_cols = ['cdp',
                 'cdi',
                 'Drag',
                 'eta',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)      
    
    for i, segment in enumerate(results.segments.values()):
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        drag_breakdown = segment.conditions.aerodynamics.drag_breakdown
        cdp = drag_breakdown.parasite.total[:,0]
        cdi = drag_breakdown.induced.total[:,0]
        cdc = drag_breakdown.compressible.total[:,0]
        cdm = drag_breakdown.miscellaneous.total[:,0]
        cde = drag_breakdown.drag_coefficient_increment[:,0]
        cd  = drag_breakdown.total[:,0]
        
        # Assemble data into temporary holding data frame
        segment_frame = pd.DataFrame(
            np.column_stack((cdp,
                             cdi,
                             cdc,
                             cdm,
                             cd)),
            columns = plot_cols[:-1], index=time
        )
        segment_frame['Segment'] = [segment.tag for i in range(len(time))]

        # Append to collecting data frame
        df = df.append(segment_frame)           
        
    # Set plot parameters
    fig = make_subplots(rows=3, cols=2)
    
    # Add traces to the figure for each value by segment
    for seg, data in df.groupby("Segment", sort=False):
        seg_name = ' '.join(seg.split("_")).capitalize()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cdp'],
            name=seg_name),
            row=1, col=1)    
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cdi'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)    
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cdc'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)         
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cdm'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)    
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cde'],
            name=seg_name,
            showlegend=False),
            row=3, col=1)           

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cd'],
            name=seg_name,
            showlegend=False),
            row=3, col=2)           
                            
    # Add subplot axis titles
    fig.update_yaxes(title_text='Parasitic CD', row=1, col=1)
    fig.update_yaxes(title_text='Induced CD', row=1, col=2)
    fig.update_yaxes(title_text='Compressibility CD', row=2, col=1)
    fig.update_yaxes(title_text='Miscellaneous CD', row=2, col=2)
    fig.update_yaxes(title_text='Excrescence CD', row=3, col=1)
    fig.update_yaxes(title_text='Total CD', row=3, col=2)

    fig.update_xaxes(title_text='Time (min)', row=3, col=1)
    fig.update_xaxes(title_text='Time (min)', row=3, col=2)

    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Drag Components'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)    

    return

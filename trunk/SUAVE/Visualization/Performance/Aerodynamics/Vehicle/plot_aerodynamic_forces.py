## @ingroup Visualization-Performance-Aerodynamics
# plot_aerodynamic_forces.py
# 
# Created:    Nov 2022, E. Botero
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
#   Aerodynamic Forces
# ---------------------------------------------------------------------- 

## @ingroup Visualization-Performance-Aerodynamics
def plot_aerodynamic_forces(results,
                            save_figure=False,
                            save_filename="Aerodynamic_Forces",
                            file_type=".png",
                            width = 1200, height = 600,
                            *args, **kwargs):
    """This plots the aerodynamic forces
    
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
    results.segments.condtions.frames
         body.thrust_force_vector
         wind.lift_force_vector
         wind.drag_force_vector
         
    Outputs:
    Plots
    
    Properties Used:
    N/A
    """
    
    # Create empty dataframe to be populated by the segment data
    plot_cols = ['Thrust',
                 'Lift',
                 'Drag',
                 'eta',
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)    

    for segment in results.segments.values():
        time   = segment.conditions.frames.inertial.time[:,0] / Units.min
        Thrust = segment.conditions.frames.body.thrust_force_vector[:,0]
        Lift   = -segment.conditions.frames.wind.lift_force_vector[:,2]
        Drag   = -segment.conditions.frames.wind.drag_force_vector[:,0]
        eta    = segment.conditions.propulsion.throttle[:,0]
        
        # Assemble data into temporary holding data frame
        segment_frame = pd.DataFrame(
            np.column_stack((Thrust,
                             Lift,
                             Drag,
                             eta)),
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
            y=data['eta'],
            name=seg_name),
            row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Lift'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Thrust'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Drag'],
            name=seg_name,
            showlegend=False),
            row=2, col=2)
                                        
    fig.update_yaxes(title_text='Throttle', row=1, col=1)
    fig.update_yaxes(title_text='Lift (N)', row=2, col=1)
    fig.update_yaxes(title_text='Thrust (N)', row=1, col=2)
    fig.update_yaxes(title_text='Drag (N)', row=2, col=2)
    
    fig.update_xaxes(title_text='Time (min)', row=2, col=1)
    fig.update_xaxes(title_text='Time (min)', row=2, col=2)        

    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = 'Aerodynamic Forces'
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename, file_type)


    return

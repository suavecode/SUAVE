## @ingroup Visualization-Performance-Energy-Common
# plot_rotor_conditions.py
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

## @ingroup Visualization-Performance-Energy-Common
def plot_rotor_conditions(results,
                              save_figure = False,
                              save_filename = "Rotor_Conditions",
                              file_type = ".png",
                              width = 1200, height = 600,
                              *args, **kwargs):
    """Plots rotor performance conditions

    Assumptions:
    None

    Deprecated MARC Mission Plots Functions

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge

    Inputs:
    results.segments.conditions.
        frames.inertial.time
        propulsion.rpm
        frames.body.thrust_force_vector
        propulsion.motor.torque
        propulsion.rotor_tip_mach

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """
    
    # determine the number of propulsor groups 
    number_of_propulsor_groups = results.segments[0].conditions.propulsion.number_of_propulsor_groups 

    for pg in range(number_of_propulsor_groups):    
        # Create empty dataframe to be populated by the segment data
        plot_cols = [
            'Thrust',
            'RPM',
            'Torque',
            'Throttle',
            'Power Coefficient',
            'Tip Mach',
            'Segment'
        ]
    
        df = pd.DataFrame(columns=plot_cols)
    
        # Get the segment-by-segment results 
        for segment in results.segments.values(): 
            time   = segment.conditions.frames.inertial.time[:,0] / Units.min 
            rpm    = segment.conditions.propulsion['propulsor_group_' + str(pg)].rotor.rpm[:,0]
            thrust = np.linalg.norm(segment.conditions.frames.body.thrust_force_vector[:,:],axis=1)
            torque = segment.conditions.propulsion['propulsor_group_' + str(pg)].motor.torque[:,0]
            tm     = segment.conditions.propulsion['propulsor_group_' + str(pg)].rotor.tip_mach[:,0]
            Cp     = segment.conditions.propulsion['propulsor_group_' + str(pg)].rotor.power_coefficient[:,0]
            eta    = segment.conditions.propulsion['propulsor_group_' + str(pg)].throttle[:,0]
    
            # Assemble the data into temporary holding dataframe
    
            segment_frame = pd.DataFrame(
                np.column_stack((
                    thrust,
                    rpm,
                    torque,
                    eta,
                    Cp,
                    tm
                )),
                columns=plot_cols[:-1], index=time
            )
    
            segment_frame['Segment'] = [segment.tag for i in range(len(time))]
    
            # Append to collecting dataframe
    
            df = df.append(segment_frame)
    
        # Set plot parameters
    
        fig = make_subplots(rows=3, cols=2) 
        # Add traces to the figure for each value by segment
    
        for seg, data in df.groupby("Segment", sort=False):
            seg_name = ' '.join(seg.split("_")).capitalize()
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Thrust'],
                name=seg_name),
                row=1, col=1)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RPM'],
                name=seg_name,
                showlegend=False),
                row=2, col=1)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Torque'],
                name=seg_name,
                showlegend=False),
                row=3, col=1)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Throttle'],
                name=seg_name,
                showlegend=False),
                row=1, col=2)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Power Coefficient'],
                name=seg_name,
                showlegend=False),
                row=2, col=2)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Tip Mach'],
                name=seg_name,
                showlegend=False),
                row=3, col=2)
    
        # Add subplot axis titles 
        fig.update_yaxes(title_text='Thrust (N)', row=1, col=1)
        fig.update_yaxes(title_text='RPM', row=2, col=1)
        fig.update_yaxes(title_text='Torque (N-m)', row=3, col=1)
        fig.update_yaxes(title_text='Throttle', row=1, col=2)
        fig.update_yaxes(title_text='Power Coefficient', row=2, col=2)
        fig.update_yaxes(title_text='Tip Mach No.', row=3, col=2)
    
        fig.update_xaxes(title_text='Time (min)', row=3, col=1)
        fig.update_xaxes(title_text='Time (min)', row=3, col=2)
    
        # Set overall figure layout style and legend title
    
        fig.update_layout(
            width=width, height=height,
            legend_title_text='Segment',
            title_text = 'Propulsor Group ' + str(pg) + ': Rotor Conditions'
        )
    
        fig = plot_style(fig)
        fig.show()
    
        if save_figure:
            save_plot(fig, save_filename, file_type)

    return

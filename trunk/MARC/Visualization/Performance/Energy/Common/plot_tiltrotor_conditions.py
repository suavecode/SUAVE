## @ingroup Visualization-Performance-Energy-Common
# plot_tiltrotor_conditions.py
# 
# Created:    Dec 2022, M. Clarke
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
def plot_tiltrotor_conditions(results,configs,
                              save_figure = False,
                              show_figure  = True,
                              save_filename = "Tiltrotor",
                              file_type = ".png",
                              width = 1200, height = 600,
                              *args, **kwargs):
    
    """This plots the tiltrotor conditions

    Assumptions:
    None

    Source:

    Deprecated MARC Mission Plots Functions

    Created:     
    
    results.segments.conditions.
        frames.inertial.time
        propulsion.y_axis_rotation


    Outputs:
    Plots

    Properties Used:
    N/A
    """

    # determine the number of propulsor groups 
    number_of_propulsor_groups = results.segments[0].conditions.propulsion.number_of_propulsor_groups 

    for pg in range(number_of_propulsor_groups):        

        config = configs[list(configs.keys())[0]]
        net    = config.networks[list(config.networks.keys())[0]]
        props  = net.rotors
        D      = 2 * props[list(props.keys())[0]].tip_radius
        
    
        # Create empty dataframe to be populated by the segment data
    
        plot_cols = [
            'Y_Rot',
            'Pitch',
            'Advance_Ratio',
            'Rotor_Incidence', 
            'Segment'
        ]
    
        df = pd.DataFrame(columns=plot_cols)
        
        for segment in results.segments.values(): 
    
            Vx         = segment.state.conditions.frames.inertial.velocity_vector[:,0]
            Vz         = segment.state.conditions.frames.inertial.velocity_vector[:,2] 
            body_angle = segment.state.conditions.frames.body.inertial_rotations[:,1] / Units.deg
            y_rot      = segment.conditions.propulsion['propulsor_group_' + str(pg)].y_axis_rotation[:,0] / Units.deg
            time       = segment.conditions.frames.inertial.time[:,0] / Units.min
            Vinf       = segment.conditions.freestream.velocity[:,0]
    
            thrust_vector = segment.conditions.frames.body.thrust_force_vector
            Tx = thrust_vector[:,0]
            Tz = thrust_vector[:,2]
            thrust_angle  = np.arccos(Tx / np.sqrt(Tx**2 + Tz**2))
            velocity_angle = np.arctan(-Vz / Vx)
    
            n     = segment.conditions.propulsion['propulsor_group_' + str(pg)].rotor.rpm[:,0] / 60
            J     = Vinf/(n*D)
    
            prop_incidence_angles =  thrust_angle - velocity_angle 
    
            # Assemble data into a temporary holding dataframe 
            segment_frame = pd.DataFrame(
                np.column_stack((
                    y_rot,
                    body_angle,
                    J,
                    prop_incidence_angles/Units.deg, 
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
                y=data['Y_Rot'],
                name=seg_name),
                row=1, col=1)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Pitch'],
                name=seg_name,
                showlegend=False),
                row=2, col=1)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Advance_Ratio'],
                name=seg_name,
                showlegend=False),
                row=1, col=2)
            
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Rotor_Incidence'],
                name=seg_name,
                showlegend=False),
                row=2, col=2)        
    
        fig.update_yaxes(title_text='Network Y-Axis Rotation (deg)', row=1, col=1)
        fig.update_yaxes(title_text='Aircraft Pitch', row=2, col=1)
        fig.update_yaxes(title_text='Advance Ratio (J=V/nD)', row=1, col=2)
        fig.update_yaxes(title_text='Rotor Incidence', row=2, col=2) 
        fig.update_xaxes(title_text='Time (min)', row=2, col=1)
        fig.update_xaxes(title_text='Time (min)', row=2, col=2)
    
        # Set overall figure layout style and legend title
    
        fig.update_layout(
            width=width, height=height,
            legend_title_text='Segment',
            title_text = 'Propulsor Group ' + str(pg) + ': Tilting Rotor'
        )
    
        fig = plot_style(fig)
        if show_figure:
            fig.write_html( save_filename + '.html', auto_open=True)
    
        if save_figure:
            save_plot(fig, save_filename, file_type)

    return 
 
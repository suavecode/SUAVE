## @ingroup Visualization-Performance-Mission
# plot_flight_trajectory.py
# 
# Created:    Dec 2022, M. Clarke
# Modified:   


from MARC.Core import Units
from MARC.Visualization.Performance.Common import plot_style, save_plot 
import numpy as np
import pandas as pd 
from itertools import cycle
import plotly.express as px
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
#  2D Aircraft Trajectory
# ------------------------------------------------------------------
## @ingroup Visualization-Performance-Mission
def plot_flight_trajectory(results,
                            save_figure=False,
                            save_filename="Flight_Trajectory",
                            file_type=".png",
                            width = 1200, height = 600,
                            *args, **kwargs):
    """This plots the 3D flight trajectory of the aircraft.
    Assumptions:
    None
    Source:
    None
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
    

    # ---------------------------------------------------------------------------------
    # Three Dimensional Trajectory
    # ---------------------------------------------------------------------------------
    plot_data = []       
    plot_cols = ['X',
                 'Y',
                 'Z', 
                 'Segment']

    df = pd.DataFrame(columns=plot_cols)    

    # 3d initiate 3d plot limits 
    x_min,x_max = 0,0
    y_min,y_max = 0,0
    z_min,z_max = 0,0
    
    for segment in results.segments.values():
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min
        X        = segment.conditions.frames.inertial.position_vector[:,0]
        Y        = segment.conditions.frames.inertial.position_vector[:,1]
        Z        = -segment.conditions.frames.inertial.position_vector[:,2]
        
        # update 3d plot limits 
        x_min,x_max = np.minimum(x_min,np.min(X)) ,np.maximum(x_max,np.max(X))
        y_min,y_max = np.minimum(y_min,np.min(Y)) ,np.maximum(y_max,np.max(Y))
        z_min,z_max = 0 , np.maximum(z_max,np.max(Z))
        
        segment_frame = pd.DataFrame(
        np.column_stack((X,
                         Y,
                         Z)),
        columns = plot_cols[:-1], index=time
        )

        segment_frame['Segment'] = [segment.tag for i in range(len(time))]
    
        # Append to collecting data frame
        df = df.append(segment_frame)               
    
    # Set plot parameters
    fig = make_subplots(rows=2, cols=2)

    # Setup the colors
    NS           = len(results.segments)+1
    color_list   = px.colors.sample_colorscale("inferno", [n/(NS -1) for n in range(NS)]) 
    colorcycler  = cycle(color_list)  

    # Add traces to the figure for each value by segment
    for seg, data in df.groupby("Segment", sort=False):
        segment_color = next(colorcycler)
        seg_name = ' '.join(seg.split("_")).capitalize()  
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['X'],
            name=seg_name),
            row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Y'],
            name=seg_name,
            showlegend=False),
            row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Z'],
            name=seg_name,
            showlegend=False),
            row=2, col=1)  
 
    
        # 3D Aircraft Trajectory
        aircraft_trajectory = go.Scatter3d(x=data['X'], y=data['Y'], z=data['Z'],
                                           mode='markers',
                                           name=seg_name,
                                           marker=dict(size=6,color=segment_color,opacity=0.8),
                                           line=dict(color=segment_color,width=2)) 
        
        plot_data.append(aircraft_trajectory) 
         
    fig.update_yaxes(title_text='X-Direction Distance (m)', row=1, col=1)
    fig.update_yaxes(title_text='Y-Direction Distance (m)', row=1, col=2)
    fig.update_yaxes(title_text='Z-Direction Distance (m)', row=2, col=1)  
    fig.update_yaxes(title_text='X-Direction Distance (m)', row=1, col=1) 
     
    fig.update_xaxes(title_text='Time (min)', row=1, col=2)   
    fig.update_xaxes(title_text='Time (min)', row=1, col=1) 
    fig.update_xaxes(title_text='Time (min)', row=2, col=1)  
    
    # Set overall figure layout style and legend title
    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
        title_text = '2D Flight Trajectory',
    )

    fig = plot_style(fig)
    fig.show()

    if save_figure:
        save_plot(fig, save_filename + '_2D', file_type) 
    
    # ---------------------------------------------------------------------------------
    # Three Dimensional Trajectory
    # ---------------------------------------------------------------------------------
    plot_axis = True 
    camera        = dict(up=dict(x=0, y=0, z=1), center=dict(x=-0.05, y=0, z=-0.25), eye=dict(x=-1., y=-1., z=.4))    
    fig_3d = go.Figure(data=plot_data)  
    fig_3d.update_scenes(aspectratio=dict(x = 1,y = 1,z =0.5)) 
    fig_3d.update_layout(title_text = '3D Flight Trajectory',  
                         width     = 1200,
                         height    = 900,
                         font_size = 12,  
                         scene_camera=camera, 
                         scene = dict(
                             xaxis = dict(backgroundcolor="white", gridcolor="grey", showbackground=plot_axis,
                                     zerolinecolor="grey", range=[x_min,x_max]),
                             yaxis = dict(backgroundcolor="white", gridcolor="grey", showbackground=plot_axis, 
                                     zerolinecolor="grey", range=[y_min,y_max]),
                             zaxis = dict(backgroundcolor="white",gridcolor="grey",showbackground=plot_axis,
                                     zerolinecolor="grey", range=[0,z_max])))
 
    fig_3d.show()
    if save_figure:
        save_plot(fig_3d, save_filename + '_3D', file_type)  
        
    return 
             
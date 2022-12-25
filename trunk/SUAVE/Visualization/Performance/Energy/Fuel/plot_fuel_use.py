## @ingroup Visualization-Performance-Energy-Fuel
# plot_fuel_use.py
# 
# Created:    Nov 2022, J. Smart
# Modified:    
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------  
from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Common import plot_style, save_plot

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

## @ingroup Visualization-Performance-Energy-Fuel
def plot_fuel_use(results,
                    save_figure = False,
                    save_filename = "Aircraft_Fuel_Burnt" ,
                    file_type = ".png",
                    width=800,height=500,
                    *args, **kwargs):
     

    """This plots aircraft fuel usage
    
    
    Assumptions:
    None

    Source:

    Depricated SUAVE Mission Plots Functions

    Created:    Mar 2020, M. Clarke
    Modified:   Apr 2020, M. Clarke
                Sep 2020, M. Clarke
                Apr 2021, M. Clarke
                Dec 2021, S. Claridge

  
    Inputs:
    results.segments.condtions.
        frames.inertial.time
        weights.fuel_mass
        weights.additional_fuel_mass
        weights.total_mass
        
    Outputs:
    Plots
    Properties Used:
    N/A	"""


    # Create empty data frame to be populated by the segment data 
    df = pd.DataFrame(columns=['Fuel', 'Add_Fuel', 'Tot_Fuel', 'Segment'])

    # Get the segment-by-segment results for altitude, mass, and the
    # SFC (calculated)
    prev_seg_fuel        = 0
    prev_seg_extra_fuel  = 0
    total_fuel           = 0
    additional_fuel_flag = False 
    
    for seg_idx in range(len(results.segments)): 
        segment  = results.segments[seg_idx] 
        time     = segment.conditions.frames.inertial.time[:,0] / Units.min 
        
        if "has_additional_fuel" in segment.conditions.weights and segment.conditions.weights.has_additional_fuel == True:
            additional_fuel_flag = True 
            fuel                 = segment.conditions.weights.fuel_mass[:,0]
            alt_fuel             = segment.conditions.weights.additional_fuel_mass[:,0]
        
            if seg_idx == 0:
        
                plot_fuel     = np.negative(fuel)
                plot_alt_fuel = np.negative(alt_fuel)
                total_fuel    = np.add(plot_fuel, plot_alt_fuel)
                
                # Assemble data into temporary holding data frame
            
                segment_frame = pd.DataFrame(
                np.column_stack((plot_fuel,plot_alt_fuel,total_fuel)),
                columns=['Fuel', 'Add_Fuel', 'Tot_Fuel'], index=time)
                segment_frame['Segment'] = [segment.tag for i in range(len(time))]
            
                # Append to collecting data-frame 
                df = df.append(segment_frame)
                 
        
            else:
                prev_seg_fuel       += results.segments[seg_idx-1].conditions.weights.fuel_mass[-1]
                prev_seg_extra_fuel += results.segments[seg_idx-1].conditions.weights.additional_fuel_mass[-1]
            
                current_fuel         = np.add(fuel, prev_seg_fuel)
                current_alt_fuel     = np.add(alt_fuel, prev_seg_extra_fuel) 

                # Assemble data into temporary holding data frame 
                segment_frame = pd.DataFrame(
                np.column_stack((np.negative(current_fuel) ,np.negative(current_alt_fuel ),np.negative(current_fuel + current_alt_fuel))),
                columns=['Fuel', 'Add_Fuel', 'Tot_Fuel'], index=time)
                segment_frame['Segment'] = [segment.tag for i in range(len(time))]
            
                # Append to collecting data-frame 
                df = df.append(segment_frame) 
        else: 
            initial_weight  = results.segments[0].conditions.weights.total_mass[:,0][0]
     
            fuel        = segment.conditions.weights.total_mass[:,0] 
            total_fuel  = np.negative(segment.conditions.weights.total_mass[:,0] - initial_weight ) 

            # Assemble data into temporary holding data frame 
            segment_frame = pd.DataFrame(
            np.column_stack((total_fuel)),
            columns=['Tot_Fuel'], index=time)
            segment_frame['Segment'] = [segment.tag for i in range(len(time))]
        
            # Append to collecting data-frame 
            df = df.append(segment_frame)  
                
    # Set plot parameters 
    if additional_fuel_flag: 
        fig = make_subplots(rows=3, cols=1,vertical_spacing=0.05)
    else:
        fig = make_subplots(rows=1, cols=1,vertical_spacing=0.05)

    # Add traces to the figure for each value by segment.

    for seg, data in df.groupby("Segment"):
        if additional_fuel_flag: 
            seg_name = ' '.join(seg.split("_")).capitalize()
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Fuel'],
                name=seg_name),
                          row=1, col=1)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Add_Fuel'],
                name=seg_name,
                showlegend=False),
                          row=2, col=1)
    
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Tot_Fuel'],
                name=seg_name,
                showlegend=False),
                          row=3, col=1)
        else:

            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Tot_Fuel'],
                name=seg_name,
                showlegend=False),
                          row=1, col=1)            

    # Set sublot axis titles 
    if additional_fuel_flag:
        fig.update_yaxes(title_text='Fuel (kg)', row=1, col=1)
        fig.update_yaxes(title_text='Additional Fuel (kg)', row=2, col=1)
        fig.update_yaxes(title_text='Total Fuel (kg)', row=3, col=1) 
        fig.update_xaxes(title_text='Time (min)', row=3, col=1)
    else: 
        if additional_fuel_flag:
            fig.update_yaxes(title_text='Fuel (kg)', row=1, col=1) 
            fig.update_xaxes(title_text='Time (min)', row=1, col=1)        

    # Set overall figure layout style and legend title

    fig.update_layout(
        width=width, height=height,
        legend_title_text='Segment',
    )

    # Update Figure Style and Show 
    fig = plot_style(fig)
    fig.show()

    # Optionally save the figure with kaleido import check

    if save_figure:
        save_plot(fig, save_filename, file_type)
 
    return

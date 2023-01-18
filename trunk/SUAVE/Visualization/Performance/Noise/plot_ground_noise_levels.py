## @ingroup Visualization-Performance-Noise
# plot_ground_noise_levels.py
# 
# Created:    Dec 2022, M. Clarke
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from SUAVE.Core import Units
from SUAVE.Visualization.Performance.Common import plot_style, save_plot
import numpy as np  
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 
from itertools import cycle
import plotly.express as px
import plotly
from SUAVE.Visualization.Performance.Common.post_process_noise_data import post_process_noise_data
  
## @ingroup Visualization-Performance-Noise
def plot_ground_noise_levels(results,
                            save_figure=False,
                            save_filename="Sideline Noise Levels",
                            file_type=".png",
                            width = 1200, height = 600,
                            *args, **kwargs): 
    """This plots the A-weighted Sound Pressure Level as a function of time at various aximuthal angles
    on the ground

    Assumptions:
    None

    Source:
    None

    Inputs:
    results.segments.conditions.
        frames.inertial.position_vector   - position vector of aircraft
        noise.
            total_SPL_dBA                 - total SPL (dbA)
            total_microphone_locations    - microphone locations

    Outputs:
    Plots

    Properties Used:
    N/A
    """    
    
    noise_data   = post_process_noise_data(results) 
    N_gm_y       = noise_data.N_gm_y
    SPL          = noise_data.SPL_dBA_ground_mic      
    gm           = noise_data.SPL_dBA_ground_mic_loc    
    gm_x         = gm[:,:,0]
    gm_y         = gm[:,:,1]  
    max_SPL      = np.max(SPL,axis=0)  
    

    # Setup the colors
    NS           = 10+1
    color_list   = px.colors.sample_colorscale("plasma", [n/(NS -1) for n in range(NS)]) 
    colorcycler  = cycle(color_list) 
    fig          = make_subplots(rows=1, cols=1) 
      
    for j in range(N_gm_y):  
        segment_color = next(colorcycler) 
        fig.add_trace(go.Scatter(x= gm_x[:,0],
                                 y= max_SPL[:,j],
                                 showlegend=True,
                                 mode='lines',
                                 name=r'y = ' + str(round(gm_y[0,j],1)) + r' m' ,
                                 line=dict(color=segment_color)),
                                 col=1,row=1)           
 
    fig.update_yaxes(title_text='$SPL (dBA)$', row=1, col=1)  
    fig.update_xaxes(title_text='Range (m)', row=1, col=1)   
    fig.update_layout(title_text= 'Microphone Noise') 
    if save_figure:
        save_plot(fig, save_filename, file_type)     
        
    fig.show() 
         
    return    
 


## @ingroup Visualization-Performance-Noise
# plot_flight_profile_noise_contours.py
# 
# Created:    Dec 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------   
from MARC.Core import Units 
import numpy as np  
import plotly.graph_objects as go
from MARC.Visualization.Geometry.Common.contour_surface_slice      import contour_surface_slice
from MARC.Visualization.Performance.Common.post_process_noise_data import post_process_noise_data 
 
## @ingroup Visualization-Performance-Noise
def plot_flight_profile_noise_contours(results,
                                       save_figure=False,
                                       show_figure=True,
                                       save_filename="Noise_Contour",
                                       colormap = 'jet',
                                       file_type=".png",
                                       width = 1200, height = 600,
                                       *args, **kwargs): 
    """This plots two contour surfaces of the maximum A-weighted Sound Pressure Level in the defined computational domain.
    The first contour is the that of radiated noise on level ground while the second includes relief.

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
    noise_data      = post_process_noise_data(results)  
    SPL_contour_gm  = noise_data.SPL_dBA_ground_mic       
    Aircraft_pos    = noise_data.aircraft_position       
    X               = noise_data.SPL_dBA_ground_mic_loc[:,:,0]  
    Y               = noise_data.SPL_dBA_ground_mic_loc[:,:,1]  
    Z               = noise_data.SPL_dBA_ground_mic_loc[:,:,2]  
    plot_data       = []  
    max_SPL_gm      = np.max(SPL_contour_gm,axis=0)  

    # ---------------------------------------------------------------------------
    # TWO DIMENSIONAL NOISE CONTOUR
    # ---------------------------------------------------------------------------
    min_SPL, max_SPL   = 35, 100 
    fig_2d = go.Figure(data = go.Contour(z=max_SPL_gm, x=X[:,0],  y=Y[0,:],
                                         contours=dict(
                                                  start= min_SPL,
                                                  end  = max_SPL,
                                                  size = 5),                                       
                                          colorbar=dict(
                                          title='SPL (dBA)',  
                                          titleside='right',
                                          titlefont=dict(size=14))))
    
    fig_2d.update_xaxes(title_text='West <------- Distance (m) -----> East ')
    fig_2d.update_yaxes(title_text='South <------- Distance (m) -----> North ')   
    fig_2d.update_layout(title_text= '2D Noise Contour')       

    if show_figure: 
        fig_2d.show() 
    if save_figure:
        save_filename_2d = save_filename.replace("_", " ")  + '_2D'
        fig_2d.write_image(save_filename_2d + file_type) 
        
    # ---------------------------------------------------------------------------
    # TRHEE DIMENSIONAL NOISE CONTOUR
    # --------------------------------------------------------------------------- 
    # TERRAIN CONTOUR 
    ground_contour   = contour_surface_slice(Y,X,Z,max_SPL_gm,color_scale=colormap)
    plot_data.append(ground_contour)

    # AIRCRAFT TRAJECTORY
    aircraft_trajectory = go.Scatter3d(x=Aircraft_pos[:,1], y=Aircraft_pos[:,0], z=Aircraft_pos[:,2],
                                       mode='markers',
                                marker=dict(size=6,color='black',opacity=0.8),
                                line=dict(color='black',width=2))
    plot_data.append(aircraft_trajectory)

    # Define Colorbar Bounds
    min_alt     = np.minimum(np.min(Z),0)
    max_alt     = np.maximum(np.max(Z), np.max(Aircraft_pos[:,2])) 

    camera        = dict(up=dict(x=0, y=0, z=1), center=dict(x=-0.05, y=0, z=-0.25), eye=dict(x=-1., y=-1., z=.4))    
    fig_3d = go.Figure(data=plot_data)  
    fig_3d.update_scenes(aspectratio=dict(x = 1,y = 1,z =0.5)) 
    fig_3d.update_layout(
        title_text= 'Flight_Profile_' + save_filename, 
             title_x   = 0.5,
             width     = 1200,
             height    = 900,
             font_size = 12,
             scene_zaxis_range=[min_alt,max_alt], 
             coloraxis=dict(colorscale=colormap,
                            colorbar_thickness=50,
                            colorbar_nticks=20,
                            colorbar_title_text = 'SPL (dBA)',
                            colorbar_tickfont_size=16,
                            colorbar_title_side="right",
                            colorbar_ypad= 50,
                            colorbar_len= 0.75,
                            **colorax(min_SPL, max_SPL)),
             scene_camera=camera) 
    
    if show_figure:
        fig_3d.show() 
    if save_figure:
        fig_3d.write_image(save_filename + '_3D', file_type)
         
    return        

def colorax(vmin, vmax):
    return dict(cmin=vmin,
                cmax=vmax)
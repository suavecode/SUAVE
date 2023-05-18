## @ingroup Visualization-Performance-Noise
# plot_noise_contour.py
# 
# Created:    Dec 2022, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------   
import numpy as np  
import plotly.graph_objects as go
from MARC.Core import Units
from MARC.Visualization.Geometry.Common.contour_surface_slice      import contour_surface_slice 
 
## @ingroup Visualization-Performance-Noise
def plot_3D_noise_contour(noise_data,
                       noise_level              = None ,
                       min_noise_level          = 35,  
                       max_noise_level          = 90, 
                       noise_scale_label        = None,
                       save_figure              = False,
                       show_figure              = True,
                       save_filename            = "Noise_Contour",
                       use_lat_long_coordinates = True, 
                       show_trajectory          = False,
                       show_microphones         = False,
                       colormap                 = 'jet',
                       file_type                = ".png",
                       background_color         = 'rgb(17,54,71)',
                       grid_color               = 'gray',
                       width                    = 1400, 
                       height                   = 800,
                       *args, **kwargs): 
    """This plots a 3D noise contour of a noise level using plotly

    Assumptions:
    None

    Source:
    None

    Inputs: 
       noise_data        - noise data structure 
       noise_level       - noise level (dBA, DNL, SENEL etc)
       min_noise_level   - minimal noise level 
       max_noise_level   - maximum noise level 
       noise_scale_label - noise level label 
       save_figure       - save figure 
       show_figure       - show figure 
       save_filename     - save file flag
       show_trajectory   - plot aircraft trajectory flag
       show_microphones  - show microhpone flag 

    Outputs:
       Plots

    Properties Used:
    N/A
    """   
    Aircraft_pos    = noise_data.aircraft_position      
    X               = noise_data.ground_microphone_locations[:,:,0]/Units.nmi  
    Y               = noise_data.ground_microphone_locations[:,:,1]/Units.nmi  
    Z               = noise_data.ground_microphone_locations[:,:,2]/Units.feet  
    plot_data       = []   
  
    # ---------------------------------------------------------------------------
    # TRHEE DIMENSIONAL NOISE CONTOUR
    # --------------------------------------------------------------------------- 
    # TERRAIN CONTOUR 
    ground_contour   = contour_surface_slice(Y,X,Z,noise_level,color_scale=colormap)
    plot_data.append(ground_contour)

    # GROUND MICROPHONES
    if show_microphones:
        microphones = go.Scatter3d(x        = Y.flatten(),
                                   y        = X.flatten(),
                                   z        = Z.flatten(),
                                   mode     = 'markers',
                                   marker   = dict(size=6,color='white',opacity=0.8),
                                   line     = dict(color='white',width=2))
        plot_data.append(microphones)

    # AIRCRAFT TRAJECTORY
    if show_trajectory:
        aircraft_trajectory = go.Scatter3d(x   = Aircraft_pos[:,1]/Units.nmi,
                                           y   = Aircraft_pos[:,0]/Units.nmi,
                                           z   = Aircraft_pos[:,2]/Units.feet,
                                           mode= 'markers',
                                           marker=dict(size=6,
                                                       color='black',
                                                       opacity=0.8),
                                    line=dict(color='black',width=2))
        plot_data.append(aircraft_trajectory)

    # Define Colorbar Bounds
    min_alt     = np.minimum(np.min(Z),0) 
    max_alt     = np.maximum(np.max(Z), np.max(Aircraft_pos[:,2]/Units.feet)) 
  
    fig_3d = go.Figure(data=plot_data) 

    if show_microphones or show_trajectory:
        pass
    else: 
        fig_3d.update_traces(colorbar_orientation     = 'v',
                             colorbar_thickness       = 50,
                             colorbar_nticks          = 10,
                             colorbar_title_text      = noise_scale_label,
                             colorbar_tickfont_size   = 16,
                             colorbar_title_side      = "right",
                             colorbar_ypad            = 50,
                             colorbar_len             = 0.75)
        
                         
    fig_3d.update_layout(
             title_text                             = save_filename, 
             title_x                                = 0.5,
             width                                  = width,
             height                                 = height,
             font_size                              = 12,
             scene_aspectmode                       = "manual",
             scene_aspectratio                      = dict(x=1, y=1, z=0.5),      
             scene_zaxis_range                      = [min_alt,max_alt*3],
             scene                                  = dict(xaxis_title='Latitude [nmi]',
                                                           yaxis_title='Longitude [nmi]',
                                                           zaxis_title='Elevation [ft]',
                                                           xaxis = dict(
                                                                backgroundcolor=background_color,
                                                                gridcolor="white",
                                                                showbackground=True,
                                                                zerolinecolor=grid_color,),
                                                           yaxis = dict(
                                                               backgroundcolor=background_color,
                                                               gridcolor=grid_color,
                                                               showbackground=True,
                                                               zerolinecolor="white"),
                                                           zaxis = dict(
                                                               backgroundcolor=background_color,
                                                               gridcolor=grid_color,
                                                               showbackground=True,
                                                               zerolinecolor="white",),),
             scene_camera=dict(up    = dict(x=0, y=0, z=1),
                               center= dict(x=-0.05, y=0, z=-0.20),
                               eye   = dict(x=-1.0, y=-1.0, z=.4))   
    ) 
    if show_figure:
        fig_3d.show() 
    if save_figure:
        fig_3d.write_image(save_filename, file_type)

    return        

def colorax(vmin, vmax):
    return dict(cmin=vmin,
                cmax=vmax)
 
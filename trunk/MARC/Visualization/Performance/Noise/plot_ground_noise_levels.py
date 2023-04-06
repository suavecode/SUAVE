## @ingroup Visualization-Performance-Noise
# plot_ground_noise_levels.py
# 
# Created:    Dec 2022, M. Clarke
# Modified:   

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Units
from MARC.Visualization.Performance.Common import set_axes, plot_style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 
from MARC.Visualization.Performance.Common.post_process_noise_data import post_process_noise_data
  
## @ingroup Visualization-Performance-Noise
def plot_ground_noise_levels(results,
                            save_figure=False,
                            save_filename="Sideline_Noise_Levels",
                            file_type=".png",
                            width = 12, height = 7): 
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
    

    # get plotting style 
    ps      = plot_style()  

    parameters = {'axes.labelsize': ps.axis_font_size,
                  'xtick.labelsize': ps.axis_font_size,
                  'ytick.labelsize': ps.axis_font_size,
                  'axes.titlesize': ps.title_font_size}
    plt.rcParams.update(parameters) 
      
    fig   = plt.figure(save_filename)
    fig.set_size_inches(width,height)
    axes        = fig.add_subplot(1,1,1) 
    
    # get line colors for plots 
    line_colors   = cm.inferno(np.linspace(0,0.9,N_gm_y))  
      
    for k in range(N_gm_y):    
        axes.plot(gm_x[:,0]/Units.nmi, max_SPL[:,k], marker = 'o', color = line_colors[k], label= r'mic at y = ' + str(round(gm_y[0,k],1)) + r' m' ) 
    axes.set_ylabel('SPL (dBA)')
    axes.set_xlabel('Range (nautical miles)')  
    set_axes(axes)
    axes.legend(loc='upper right')         
    if save_figure:
        plt.savefig(save_filename + ".png")    
        
    return
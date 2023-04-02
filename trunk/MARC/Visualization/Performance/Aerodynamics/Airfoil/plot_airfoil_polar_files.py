## @ingroup Plots
# Airfoil_Plots.py
#
# Created:  Mar 2021, M. Clarke
# Modified: Feb 2022, M. Clarke
#           Aug 2022, R. Erhard
#           Sep 2022, M. Clarke
#           Nov 2022, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
from MARC.Core import Units
from MARC.Visualization.Performance.Common import set_axes, plot_style
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

# ----------------------------------------------------------------------
#  Plot Airfoil Polar Files
# ----------------------------------------------------------------------  

## @ingroup Visualization-Performance
def plot_airfoil_polar_files(polar_data,
                             save_figure = False,
                             save_filename = "Airfoil_Polars",
                             file_type = ".png",
                             width = 12, height = 7):
    """This plots all airfoil polars in the list "airfoil_polar_paths" 

    Assumptions:
    None

    Source:
    None

    Inputs:
    airfoil_polar_paths   [list of strings]

    Outputs: 
    Plots

    Properties Used:
    N/A	
    """ 
  
    
    # get plotting style 
    ps      = plot_style()  
    
    parameters = {'axes.labelsize': ps.axis_font_size,
                  'xtick.labelsize': ps.axis_font_size,
                  'ytick.labelsize': ps.axis_font_size,
                  'axes.titlesize': ps.title_font_size}
    plt.rcParams.update(parameters)
    
    
    # Get raw data polars
    CL           = polar_data.lift_coefficients
    CD           = polar_data.drag_coefficients
    alpha        = polar_data.angle_of_attacks/Units.degrees
    Re_raw       = polar_data.reynolds_numbers
    n_Re         = len(polar_data.re_from_polar) 
        
     
    # get line colors for plots 
    line_colors   = cm.inferno(np.linspace(0,0.9,n_Re))     
     
    fig   = plt.figure(save_filename)
    fig.set_size_inches(width,height) 
      
    for j in range(n_Re):
        
        Re_val = str(round(Re_raw[j])/1e6)+'e6'  
        
        axes_1 = plt.subplot(2,2,1)
        axes_1.plot(alpha, CL[j,:], color = line_colors[j], marker = ps.marker, linewidth = ps.line_width, label ='Re='+Re_val)
        axes_1.set_ylabel(r'$C_l$')
        axes_1.set_xlabel(r'$\alpha$')
        set_axes(axes_1)    
        
        axes_2 = plt.subplot(2,2,2)
        axes_2.plot(alpha,CD[j,:], color = line_colors[j], marker = ps.marker, linewidth = ps.line_width, label ='Re='+Re_val) 
        axes_2.set_ylabel(r'$C_d$')
        axes_2.set_xlabel(r'$\alpha$')
        set_axes(axes_2)  
        
        axes_3 = plt.subplot(2,2,3)
        axes_3.plot(CL[j,:],CD[j,:], color = line_colors[j], marker = ps.marker, linewidth = ps.line_width, label ='Re='+Re_val)
        axes_3.set_xlabel('$C_l$')
        axes_3.set_ylabel(r'$C_d$')
        set_axes(axes_3) 
    
        axes_4 = plt.subplot(2,2,4)
        axes_4.plot(alpha, CL[j,:]/CD[j,:], color = line_colors[j], marker = ps.marker, linewidth = ps.line_width, label ='Re='+Re_val) 
        axes_4.set_ylabel(r'$Cl/Cd$')
        axes_4.set_xlabel(r'$\alpha$')
        set_axes(axes_4)   
     
    # set title of plot 
    title_text    = 'Airfoil Polars'      
    fig.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return
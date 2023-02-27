## @ingroup Plots
# plot_airfoil_polars.py
#
# Created: Feb 2023, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 

from MARC.Core import Units
from MARC.Visualization.Performance.Common import set_axes, plot_style
import matplotlib.pyplot as plt 

# ----------------------------------------------------------------------
#  Plot Airfoil Polar Files
# ----------------------------------------------------------------------  

## @ingroup Visualization-Performance
def plot_airfoil_polars(polar_data,
                        save_figure = False, 
                        save_filename = "Airfoil_Polars",
                        file_type = ".png",
                        width = 12, height = 7):
    """This plots all the airfoil polars of a specfic airfoil

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
 
    # Get raw data polars
    CL           = polar_data.cl[0]
    CD           = polar_data.cd[0]
    CM           = polar_data.cm[0]
    alpha        = polar_data.AoA[0]/Units.degrees
    Re_raw       = polar_data.Re[0]  
       
    Re_val = str(round(Re_raw[0])/1e6)+'e6' 
    
    # get plotting style 
    ps      = plot_style()  

    parameters = {'axes.labelsize': ps.axis_font_size,
                  'xtick.labelsize': ps.axis_font_size,
                  'ytick.labelsize': ps.axis_font_size,
                  'axes.titlesize': ps.title_font_size}
    plt.rcParams.update(parameters)
      
    fig   = plt.figure()
    fig.set_size_inches(width,height) 
               
    axes_1 = plt.subplot(2,2,1)
    axes_1.plot(alpha, CL, color = ps.color, marker = ps.marker, linewidth = ps.line_width, label = 'Re = '+Re_val)
    axes_1.set_ylabel(r'$C_l$')
    set_axes(axes_1)    
    
    axes_2 = plt.subplot(2,2,2)
    axes_2.plot(alpha, CD, color = ps.color, marker = ps.marker, linewidth = ps.line_width)
    axes_2.set_ylabel(r'$C_d$')
    set_axes(axes_2) 

    axes_3 = plt.subplot(2,2,3)
    axes_3.plot(alpha, CM, color = ps.color, marker = ps.marker, linewidth = ps.line_width)
    axes_3.set_xlabel('AoA [deg]') 
    axes_3.set_ylabel(r'$C_m$')
    set_axes(axes_3) 
    
    axes_4 = plt.subplot(2,2,4)
    axes_4.plot(alpha, CL/CD, color = ps.color, marker = ps.marker, linewidth = ps.line_width)
    axes_4.set_xlabel('AoA [deg]')
    axes_4.set_ylabel(r'Cl/Cd')
    set_axes(axes_4) 
            
    if save_figure:
        plt.savefig(save_filename + file_type)   
    return    
     
     
     
     
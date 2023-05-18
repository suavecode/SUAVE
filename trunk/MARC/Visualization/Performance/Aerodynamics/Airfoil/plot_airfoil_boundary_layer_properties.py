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
#  Plot Airfoil Boundary Layer Properties
# ---------------------------------------------------------------------- 

## @ingroup Visualization-Performance
def plot_airfoil_boundary_layer_properties(ap,
                                           save_figure = False,
                                           show_legend = False,
                                           file_type = ".png",
                                           save_filename = 'Airfoil_with_Boundary_Layers', 
                                           width = 12, height = 7):
    """Plots viscous distributions
    
    Assumptions:
    None
    
    Source: 
    None
                                                     
    Inputs:
        ap     : data stucture of airfoil boundary layer properties  
                                                                           
    Outputs:
        Figures of quantity distributions
    
    Properties Used:
    N/A
    """      
    # get plotting style 
    ps      = plot_style()  
    
    plot_quantity(ap, ap.Ue_Vinf, r'U_e/U_inf$'  ,'Inviscid Edge Velocity',0,3,file_type,show_legend,save_figure,width,height) 
    plot_quantity(ap, ap.H,  r'H'  ,'Kinematic Shape Parameter',-1,10,file_type,show_legend,save_figure,width,height) 
    plot_quantity(ap, ap.delta_star, r'delta*' ,'Displacement Thickness',-0.01,0.1 ,file_type,show_legend,save_figure,width,height) 
    plot_quantity(ap, ap.delta   , r'delta' ,'Boundary Layer Thickness',-0.01,0.1 ,file_type,show_legend,save_figure,width,height) 
    plot_quantity(ap, ap.theta, r'theta' ,'Momentum Thickness',-0.001, 0.015,file_type,show_legend,save_figure,width,height) 
    plot_quantity(ap, ap.cf, r'c_f'  ,   'Skin Friction Coefficient',-0.1,1,file_type,show_legend,save_figure,width,height) 
    plot_quantity(ap, ap.Re_theta,  r'Re_theta'  ,'Theta Reynolds Number',-2E2,1E3,file_type,show_legend,save_figure,width,height)  
 
    n_cpts   = len(ap.AoA[:,0])
    n_cases  = len(ap.AoA[0,:]) 

    # create array of colors for difference reynolds numbers        
    blues = cm.winter(np.linspace(0,0.9,n_cases))     
    reds  = cm.autumn(np.linspace(0,0.9,n_cases))   

    fig_0   = plt.figure(save_filename)
    fig_0.set_size_inches(width,height)
    
    for i in range(n_cpts):   
        for j in range(n_cases):  
            axes_0 = plt.subplot(1,1,1)
            
            axes_0.plot(ap.x[i,j], ap.y[i,j], color = blues[j], marker = ps.marker, linewidth = ps.line_width )
            axes_0.plot(ap.x[i,j][:-1], ap.y_bl[i,j], color = reds[j], marker = ps.marker, linewidth = ps.line_width ) 
            set_axes(axes_0)    
   
    # set title of plot 
    title_text    = 'Airfoil with Boundary Layers'  
    fig_0.suptitle(title_text)
    
    if save_figure:
        plt.savefig(save_filename + file_type)   
 
    return    
 
# ----------------------------------------------------------------------
#  Plot Quantity
# ----------------------------------------------------------------------  

## @ingroup Visualization-Performance
def plot_quantity(ap, q, qaxis, qname,ylim_low,ylim_high,file_type,show_legend,save_figure,width,height) :
    """Plots a quantity q over lower/upper/wake surfaces
    
    Assumptions:
    None
    
    Source: 
    None
                                                     
    Inputs:
       ap        : data stucture of airfoil boundary layer properties  
       q         : vector of values to plot, on all points (wake too if present)
       qaxis     : name of quantity, for axis labeling
       qname     : name of quantity, for title labeling
                                                                           
    Outputs:
       Figure showing q versus x
    
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
    
    n_cpts   = len(ap.AoA[:,0])
    n_cases  = len(ap.AoA[0,:]) 
    
    fig   = plt.figure(qname.replace(" ", "_"))
    fig.set_size_inches(width,height) 
    axis  = fig.add_subplot(1,1,1)   
    
    # get line colors for plots 
    line_colors   = cm.inferno(np.linspace(0,0.9,n_cases))      
    
    for i in range(n_cpts):   
        for j in range(n_cases): 
            case_label = 'AoA: ' + str(round(ap.AoA[i,j]/Units.degrees, 2)) + ', Re: ' + str(ap.Re[i,j]) 
            axis.plot( ap.x[i,j], q[i,j], color = line_colors[j], marker = ps.marker, linewidth = ps.line_width,  label =case_label)  
            axis.set_ylim([ylim_low,ylim_high]) 
     
    if show_legend:
        leg =  fig.legend(bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol = 5) 
        
        # Adjusting the sub-plots for legend 
        fig.subplots_adjust(top=0.75)
        
    # set title of plot 
    title_text    = qname    
    fig.suptitle(title_text)
            
    if save_figure:
        plt.savefig(qname.replace(" ", "_") + file_type) 
          
    return  
  
   
           
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
from SUAVE.Core import Units 
import numpy as np 
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly

# ----------------------------------------------------------------------
#  Plot Airfoil Boundary Layer Properties
# ---------------------------------------------------------------------- 

## @ingroup Plots-Performance
def plot_airfoil_boundary_layer_properties(ap,show_legend = False ):
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
    
    plot_quantity(ap, ap.Ue_Vinf, r'$U_{e}/U_{inv}}$'  ,'Inviscid Edge Velocity',show_legend) 
    plot_quantity(ap, ap.H,  r'$H$'  ,'Kinematic Shape Parameter',show_legend) 
    plot_quantity(ap, ap.delta_star, r'$\delta*$' ,'Displacement Thickness',show_legend) 
    plot_quantity(ap, ap.delta   , r'$\delta$' ,'Boundary Layer Thickness',show_legend) 
    plot_quantity(ap, ap.theta, r'$\theta$' ,'Momentum Thickness',show_legend) 
    plot_quantity(ap, ap.cf, r'$c_f $'  ,   'Skin Friction Coefficient',show_legend) 
    plot_quantity(ap, ap.Re_theta,  r'$Re_{\theta}$'  ,'Theta Reynolds Number',show_legend) 
    

    fig = make_subplots(rows=1, cols=1)
    n_cpts   = len(ap.AoA[:,0])
    n_cases  = len(ap.AoA[0,:]) 

    # create array of colors for difference reynolds numbers         
    b = plotly.colors.get_colorscale('blues')
    r = plotly.colors.get_colorscale('reds')
    blues = px.colors.n_colors(b[0][1], b[-1][1], n_cases,colortype='rgb')
    reds  = px.colors.n_colors(r[0][1], r[-1][1], n_cases,colortype='rgb')
    
    for i in range(n_cpts):   
        for j in range(n_cases): 
            case_label = 'AoA: ' + str(round(ap.AoA[i,j]/Units.degrees, 2)) + ', Re: ' + str(ap.Re[i,j]) 
            
            fig.add_trace(go.Scatter(x=ap.x[i,j],y=ap.y[i,j],showlegend=show_legend,mode='lines',name=case_label,line=dict(color=blues[j])))
            fig.add_trace(go.Scatter(x=ap.x[i,j],y=ap.y_bl[i,j],showlegend=show_legend,mode='markers + lines',name=case_label,marker=dict(size = 3, symbol = 'circle',color = reds[j])))


    fig.update_layout(
        title='Airfoil with Boundary Layers',
        xaxis_title=r'$y$',
        yaxis_title=r'$x$',
    )        
    
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
      )              
    
    fig.show()

    return    
 
# ----------------------------------------------------------------------
#  Plot Quantity
# ----------------------------------------------------------------------  

## @ingroup Plots-Performance
def plot_quantity(ap, q, qaxis, qname,show_legend=True):
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

    n_cpts   = len(ap.AoA[:,0])
    n_cases  = len(ap.AoA[0,:]) 
    fig = make_subplots(rows=1, cols=1)
    
    for i in range(n_cpts):   
        for j in range(n_cases): 
            case_label = 'AoA: ' + str(round(ap.AoA[i,j]/Units.degrees, 2)) + ', Re: ' + str(ap.Re[i,j]) 
            fig.add_trace(go.Scatter(x=ap.x[i,j],y=q[i,j],showlegend=show_legend,mode='markers + lines',name=case_label,marker=dict(size = 5, symbol = 'circle')))

        
    fig.update_layout(
        title=qname,
        xaxis_title=qaxis,
        yaxis_title=r'$x$',
    )    
          
          
    fig.show()
    
    return  
 
# ----------------------------------------------------------------------
#  Plot Airfoil Surface Forces
# ----------------------------------------------------------------------  
 
## @ingroup Plots-Performance
def plot_airfoil_surface_forces(ap):  
    """ This plots the forces on an airfoil surface
    
        Assumptions:
        None
        
        Inputs: 
        ap       - data stucture of airfoil boundary layer properties and polars 
         
        Outputs: 
        None 
        
        Properties Used:
        N/A
        """        
    
    # determine dimension of angle of attack and reynolds number 
    n_cpts   = len(ap.AoA[:,0])
    n_cases  = len(ap.AoA[0,:])    

    for i in range(n_cpts):     
        for j in range(n_cases): 
            dx_val = ap.normals[i,j,:-1,0]*abs(ap.cp[i,j,:])*0.5
            dy_val = ap.normals[i,j,:-1,1]*abs(ap.cp[i,j,:])*0.5    
            
            fig = ff.create_quiver(ap.x[i,j,:-1], ap.y[i,j,:-1], dx_val, dy_val,showlegend=False)
            
            fig.add_trace(go.Scatter(x=ap.x[0,0,:],y=ap.y[0,0,:],showlegend=False))
            
            label =  '_AoA_' + str(round(ap.AoA[i,j]/Units.degrees,2)) + '_deg_Re_' + str(round(ap.Re[i,j]/1000000,2)) + 'E6'
            fig.update_layout(
                title={'text': 'Airfoil_Pressure_Normals' + label})
            fig.update_yaxes(
                scaleanchor = "x",
                scaleratio = 1,
              )            

    
    fig.show()
        
    return   

# ----------------------------------------------------------------------
#  Plot Airfoil Polar Files
# ----------------------------------------------------------------------  

## @ingroup Plots-Performance
def plot_airfoil_polar_files(polar_data, save_figure = False, save_filename = "Airfoil_Polars", file_type = ".png"):
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
 
    # Get raw data polars
    CL           = polar_data.lift_coefficients
    CD           = polar_data.drag_coefficients
    alpha        = polar_data.angle_of_attacks
    Re_raw       = polar_data.reynolds_numbers
    n_Re         = len(polar_data.re_from_polar)
    b            = plotly.colors.get_colorscale('blues')
    cols         = px.colors.n_colors(b[3][1], b[-1][1], n_Re,colortype='rgb')
        
    fig = make_subplots(rows=n_Re, cols=4)
      
    for j in range(n_Re):
        
        # Plotly is 1 indexed for subplots
        jj = j+1
        
        Re_val = str(round(Re_raw[j])/1e6)+'e6'
        
        fig.add_trace(go.Scatter(x=alpha/Units.degrees,y=CL[j,:],showlegend=True,mode='lines',name='Re='+Re_val,line=dict(color=cols[j])),col=1,row=jj)
        fig.add_trace(go.Scatter(x=alpha/Units.degrees,y=CD[j,:],showlegend=False,mode='lines',name='Re='+Re_val,line=dict(color=cols[j])),col=2,row=jj)
        fig.add_trace(go.Scatter(x=CL[j,:],y=CD[j,:],showlegend=False,mode='lines',name='Re='+Re_val,line=dict(color=cols[j])),col=3,row=jj)
        fig.add_trace(go.Scatter(x=alpha/Units.degrees,y=CL[j,:]/CD[j,:],showlegend=False,mode='lines',name='Re='+Re_val,line=dict(color=cols[j])),col=4,row=jj)        
        fig.update_yaxes(title_text='$C_l$', row=jj, col=1)
        fig.update_yaxes(title_text='$C_d$', row=jj, col=2)      
        fig.update_yaxes(title_text='$C_d$', row=jj, col=3)
        fig.update_yaxes(title_text='$Cl/Cd$', row=jj, col=4)            
    
    fig.update_xaxes(title_text='AoA [deg]', row=jj, col=1)
    fig.update_xaxes(title_text='AoA [deg]', row=jj, col=2)     
    fig.update_xaxes(title_text='Cl', row=jj, col=3)
    fig.update_xaxes(title_text='AoA [deg]', row=jj, col=4)        
        
    
    fig.update_layout(title_text=save_filename)    
    
    #if save_figure:
        #plt.savefig(save_filename.replace("_", " ") + file_type) 
        
    fig.show()
        
    return
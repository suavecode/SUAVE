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
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly

# ----------------------------------------------------------------------
#  Plot Airfoil Boundary Layer Properties
# ---------------------------------------------------------------------- 

## @ingroup Visualization-Performance
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

## @ingroup Visualization-Performance
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
  
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
from MARC.Visualization.Performance.Common import plot_style, save_plot
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly 

# ----------------------------------------------------------------------
#  Plot Airfoil Polar Files
# ----------------------------------------------------------------------  

## @ingroup Visualization-Performance
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
        
    
    fig.update_layout(title_text= 'Airfoil Polars')    

    fig = plot_style(fig)    
    if save_figure:
        save_plot(fig, save_filename, file_type)     
        
    fig.show()
        
    return
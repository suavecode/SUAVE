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
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly 
# ----------------------------------------------------------------------
#  Plot Airfoil Surface Forces
# ----------------------------------------------------------------------  
 
## @ingroup Visualization-Performance
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
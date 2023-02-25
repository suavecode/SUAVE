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
import MARC
from MARC.Core import Units 
import numpy as np 
import matplotlib.pyplot as plt   

# ----------------------------------------------------------------------
#  Plot Airfoil Surface Forces
# ----------------------------------------------------------------------  
 
## @ingroup Visualization-Performance
def plot_airfoil_surface_forces(ap, save_figure = False , arrow_color = 'red',save_filename = 'Airfoil_Cp_Distribution', show_figure = True, file_type = ".png"):  
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
    n_cpts   = len(ap.AoA)
    nAoA     = len(ap.AoA[0])
    n_pts    = len(ap.x[0,0,:])- 1 
     

    for i in range(n_cpts):     
        for j in range(nAoA): 
            label =  '_AoA_' + str(round(ap.AoA[i][j]/Units.degrees,2)) + '_deg_Re_' + str(round(ap.Re[i][j]/1000000,2)) + 'E6'
            fig   = plt.figure('Airfoil_Pressure_Normals' + label )
            axis = fig.add_subplot(1,1,1) 
            axis.plot(ap.x[0,0,:], ap.y[0,0,:],'k-')   
            for k in range(n_pts):
                dx_val = ap.normals[i,j,k,0]*abs(ap.cp[i,j,k])*0.1
                dy_val = ap.normals[i,j,k,1]*abs(ap.cp[i,j,k])*0.1
                if ap.cp[i,j,k] < 0:
                    plt.arrow(x= ap.x[i,j,k], y=ap.y[i,j,k] , dx= dx_val , dy = dy_val , 
                              fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
                else:
                    plt.arrow(x= ap.x[i,j,k]+dx_val , y= ap.y[i,j,k]+dy_val , dx= -dx_val , dy = -dy_val , 
                              fc=arrow_color, ec=arrow_color,head_width=0.005, head_length=0.01 )   
    
    
    return   


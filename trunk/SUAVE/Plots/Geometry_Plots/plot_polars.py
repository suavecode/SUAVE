## @ingroup Plots-Geometry_Plots
# plot_polars.py
# 
# Created:  May 2021, R. Erhard

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import matplotlib.pyplot as plt   
import numpy as np
import os
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.compute_airfoil_polars \
     import compute_airfoil_polars

def plot_polars(airfoil_polar_paths, line_color = 'k-', raw_polars = True, 
                save_figure = False, save_filename = "Airfoil_Polars", file_type = ".png"):
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
    if raw_polars:
        # Plot the raw data polars
        airfoil_polar_data = import_airfoil_polars(airfoil_polar_paths)
        CL = airfoil_polar_data.lift_coefficients
        CD = airfoil_polar_data.drag_coefficients
        
        shape      = np.shape(CL)
        n_airfoils = shape[0]
        n_Re       = shape[1]

    else:
        # Plot surrogate polars
        plot_surrogate_polars(airfoil_polar_paths)
            
    

    for i in range(n_airfoils):
        airfoil_name = os.path.basename(airfoil_polar_paths[i][0])
        
        # plot all Reynolds number polars for ith airfoil
        fig  = plt.figure(save_filename +'_'+ str(i))
        fig.set_size_inches(10, 4)
        axes = plt.subplot(1,1,1)
        axes.set_title(airfoil_name)            
        for j in range(n_Re):
            Re_val = 'Re_'+str(j)
            axes.plot(CD[i,j,:], CL[i,j,:], label=Re_val)
            
        axes.set_xlabel('$C_D$')  
        axes.set_ylabel('$C_L$')  
        axes.legend()
        
        if save_figure:
            plt.savefig(name + file_type)   
            
    return
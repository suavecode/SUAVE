## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# generate_airfoil_transition.py
# 
# Created:  Mar 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry 
from SUAVE.Plots.Geometry import plot_airfoil
import numpy as np
import os

def generate_interpolated_airfoils(a1, a2, nairfoils, npoints=200, save_filename="Transition"):
    """ Takes in two airfoils, interpolates between their coordinates to generate new
    airfoil geometries and saves new airfoil files.
    
    Assumptions: Linear geometric transition between airfoils
    
    Source: None
    
    Inputs:
    a1                 first airfoil                                [ airfoil ]
    a2                 second airfoil                               [ airfoil ]
    nairfoils          number of airfoils                           [ unitless ]
    
    """
    
    # import airfoil geometry for the two airfoils
    airfoil_geo_files = [a1, a2]
    a1_name           = os.path.basename(a1)
    a2_name           = os.path.basename(a2)
    a_geo = import_airfoil_geometry(airfoil_geo_files,npoints)
    
    # identify x and y coordinates of the two airfoils
    x_upper = a_geo.x_upper_surface
    y_upper = a_geo.y_upper_surface
    x_lower = a_geo.x_lower_surface
    y_lower = a_geo.y_lower_surface
    
    # identify points on airfoils to interpolate between
    yairfoils_upper  = np.array(y_upper).T
    yairfoils_lower  = np.array(y_lower).T
    xairfoils_upper  = np.array(x_upper).T
    xairfoils_lower  = np.array(x_lower).T

    # for each point around the airfoil, interpolate between the two given airfoil coordinates
    z = np.linspace(0,1,nairfoils)
    
    y_u_lb = yairfoils_upper[:,0]
    y_u_ub = yairfoils_upper[:,1]
    y_l_lb = yairfoils_lower[:,0]
    y_l_ub = yairfoils_lower[:,1]        
    
    x_u_lb = xairfoils_upper[:,0]
    x_u_ub = xairfoils_upper[:,1]
    x_l_lb = xairfoils_lower[:,0]
    x_l_ub = xairfoils_lower[:,1]    
    
    # broadcasting interpolation
    y_n_upper = (z[None,...] * (y_u_ub[...,None] - y_u_lb[...,None]) + (y_u_lb[...,None])).T
    y_n_lower = (z[None,...] * (y_l_ub[...,None] - y_l_lb[...,None]) + (y_l_lb[...,None])).T
    x_n_upper = (z[None,...] * (x_u_ub[...,None] - x_u_lb[...,None]) + (x_u_lb[...,None])).T
    x_n_lower = (z[None,...] * (x_l_ub[...,None] - x_l_lb[...,None]) + (x_l_lb[...,None])).T
    
    
    # save new airfoil geometry files:
    
    new_files = {'a_{}'.format(i+1): [] for i in range(nairfoils-2)}
    airfoil_files = []

    for k in range(nairfoils-2):
        # create new files and write title block for each new airfoil
        title_block     = "Airfoil Transition "+str(k+1)+" between "+a1_name+" and "+a2_name+"\n 61. 61.\n\n"
        file            ='a_'+str(k+1)
        new_files[file] = open(save_filename + str(k+1) +".txt", "w+")
        new_files[file].write(title_block)
        
        y_n_u = np.reshape(y_n_upper[k+1],(npoints//2,1))
        y_n_l = np.reshape(y_n_lower[k+1],(npoints//2,1))
        x_n_u = np.reshape(x_n_upper[k+1],(npoints//2,1))
        x_n_l = np.reshape(x_n_lower[k+1],(npoints//2,1))
        
        airfoil_files.append(new_files[file].name)
        upper_data = np.append(x_n_u, y_n_u,axis=1)
        lower_data = np.append(x_n_l, y_n_l,axis=1)

        # write lines to files
        for lines in upper_data: #upper_data[file]:
            line = str(lines[0]) + " " + str(lines[1]) + "\n"
            new_files[file].write(line)
            
        new_files[file].write("\n")
        
        for lines in lower_data: #[file]:
            line = str(lines[0]) + " " + str(lines[1]) + "\n"
            new_files[file].write(line)
            
        new_files[file].close()
        
    # plot new and original airfoils:
    airfoil_files.insert(0,a1)
    airfoil_files.append(a2)
    plot_airfoil(airfoil_files,overlay=True)
    plot_airfoil(airfoil_files,overlay=False)
    
    return new_files

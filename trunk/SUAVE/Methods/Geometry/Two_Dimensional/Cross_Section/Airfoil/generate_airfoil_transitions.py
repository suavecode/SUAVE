## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# compute_airfoil_transition.py
# 
# Created:  Mar 2021, R. Erhard
# Modified: 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry import import_airfoil_geometry 
from SUAVE.Plots.Geometry_Plots import plot_airfoil
import numpy as np

def generate_airfoil_transition(a1, a2, space, nairfoils, save_file=False, save_filename="Transition", npts=50):
    """ Takes in two airfoils and their locations and interpolates
    between them to establish a transition region.
    
    Assumptions: Linear geometric transition between airfoils
    
    Source: None
    
    Inputs:
    a1                 first airfoil                                [ airfoil ]
    a2                 second airfoil                               [ airfoil ]
    y1                 spanwise location of first airfoil           [ m ]
    y2                 spanwise location of second airfoil          [ m ]
    n                  number of transitional airfoils              [ unitless ]
    space              spanwise distance for transition to occur    [ m ]
    
    """
    
    # import airfoil geometry for the two airfoils
    airfoil_geo_files = [a1, a2]
    a_geo = import_airfoil_geometry(airfoil_geo_files,npts)
    
    # identify x and y coordinates of the two airfoils
    x_upper = a_geo.x_upper_surface
    y_upper = a_geo.y_upper_surface
    x_lower = a_geo.x_lower_surface
    y_lower = a_geo.y_lower_surface
    
    # setup spanwise stations for transitional airfoils
    z = np.linspace(0,space,nairfoils)
    zairfoils = np.array([0,space])
    
    # identify points on airfoils to interpolate between
    yairfoils_upper  = np.array(y_upper).T
    yairfoils_lower  = np.array(y_lower).T
    xairfoils_upper  = np.array(x_upper).T
    xairfoils_lower  = np.array(x_lower).T

    
    # generate new upper and lower airfoil data for transitional airfoils
    upper_data = {'a_{}'.format(i+1): [] for i in range(nairfoils-2)}
    lower_data = {'a_{}'.format(i+1): [] for i in range(nairfoils-2)}
    
    # for each point around the airfoil, interpolate between the two given airfoils
    for i in range(npts):
        y_n_upper = np.interp(z, zairfoils, yairfoils_upper[i])
        y_n_lower = np.interp(z, zairfoils, yairfoils_lower[i])
        x_n_upper = np.interp(z, zairfoils, xairfoils_upper[i])
        x_n_lower = np.interp(z, zairfoils, xairfoils_lower[i])
        
        # save interpolated data corresponding to each new airfoil
        for k in range(nairfoils-2):
            af = 'a_'+str(k+1)
            upper_data[af].append(" %f %f\n" %(x_n_upper[k+1], y_n_upper[k+1])) 
            lower_data[af].append(" %f %f\n" %(x_n_lower[k+1], y_n_lower[k+1])) 
            
            
    # save new airfoil geometry files:
    if save_file:
        new_files = {'a_{}'.format(i): [] for i in range(nairfoils-2)}
        airfoil_names = []

        for k in range(nairfoils-2):
            # create new files and write title block for each
            title_block     = "Airfoil Transition "+str(k+1)+" between"+a1+"and"+a2+"\n 61. 61.\n\n"
            file            ='a_'+str(k+1)
            new_files[file] = open(save_filename + str(k+1) +".txt", "w+")
            new_files[file].write(title_block)
            
            airfoil_names.append(new_files[file].name)
    
            # write lines to files
            for lines in upper_data[file]:
                new_files[file].write(lines)
            new_files[file].write("\n")
            for lines in lower_data[file]:
                new_files[file].write(lines)
            new_files[file].close()
        
    # plot the transition:
    airfoil_names.insert(0,a1)
    airfoil_names.append(a2)
    plot_airfoil(airfoil_names,overlay=True)
    plot_airfoil(airfoil_names,overlay=False)
      
    return

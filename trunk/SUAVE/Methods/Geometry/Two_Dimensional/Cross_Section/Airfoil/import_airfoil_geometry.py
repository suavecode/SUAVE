## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
# import_airfoil_geometry .py
# 
# Created:  Mar 2019, M. Clarke
#           Mar 2020, M. Clarke
#           Apr 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units
import numpy as np
import scipy.interpolate as interp

## @ingroup Methods-Geometry-Two_Dimensional-Cross_Section-Airfoil
def  import_airfoil_geometry(airfoil_geometry_files):
    """This imports an airfoil geometry from a text file  and stores
    the coordinates of upper and lower surfaces as well as the mean
    camberline
    
    Assumptions:
    Airfoil file in Lednicer format

    Source:
    None

    Inputs:
    airfoil_geometry_files   <list of strings>

    Outputs:
    airfoil_data.
        thickness_to_chord 
        x_coordinates 
        y_coordinates  

    Properties Used:
    N/A
    """      
 
    num_airfoils = len(airfoil_geometry_files)
    # unpack      

    airfoil_data                    = Data()
    airfoil_data.x_coordinates      = []
    airfoil_data.y_coordinates      = []
    airfoil_data.thickness_to_chord = []
    airfoil_data.camber_coordinates = []
    airfoil_data.x_upper_surface    = []
    airfoil_data.x_lower_surface    = []
    airfoil_data.y_upper_surface    = []
    airfoil_data.y_lower_surface    = []
    
    for i in range(num_airfoils):  
        # Open file and read column names and data block
        f = open(airfoil_geometry_files[i]) 
                
        # Ignore header
        for header_line in range(3):
            f.readline()     

        data_block = f.readlines()
        f.close() 
        
        x_up_surf = []
        y_up_surf = []
        x_lo_surf = []
        y_lo_surf = []
        
        # Loop through each value: append to each column
        upper_surface_flag = True
        for line_count , line in enumerate(data_block): 
            #check for blank line which signifies the upper/lower surface division 
            line_check = data_block[line_count].strip()
            if line_check == '':
                upper_surface_flag = False
                continue
            if upper_surface_flag:
                x_up_surf.append(float(data_block[line_count].strip().split()[0])) 
                y_up_surf.append(float(data_block[line_count].strip().split()[1])) 
            else:                              
                x_lo_surf.append(float(data_block[line_count].strip().split()[0])) 
                y_lo_surf.append(float(data_block[line_count].strip().split()[1]))             
        
        # determine the thickness to chord ratio - note that the upper and lower surface
        # may be of different lenghts so initial interpolation is required 
        # x coordinates
        x_up_surf_new = np.array(x_up_surf)     
        arrx          = np.array(x_lo_surf) 
        arrx_interp   = interp.interp1d(np.arange(arrx.size),arrx)
        x_lo_surf_new = arrx_interp(np.linspace(0,arrx.size-1,x_up_surf_new.size)) 
        
        # y coordinates 
        y_up_surf_new = np.array(y_up_surf)  
        arry          = np.array(y_lo_surf)
        arry_interp   = interp.interp1d(np.arange(arry.size),arry)
        y_lo_surf_new = arry_interp(np.linspace(0,arry.size-1,y_up_surf_new.size)) 
         
        # compute thickness, camber and concatenate coodinates 
        thickness     = y_up_surf_new - y_lo_surf_new
        camber        = y_lo_surf_new + thickness/2 
        x_data        = np.concatenate([x_up_surf_new,x_lo_surf_new[::-1]])
        y_data        = np.concatenate([y_up_surf_new,y_lo_surf_new[::-1]]) 
        
        airfoil_data.thickness_to_chord.append(np.max(thickness))    
        airfoil_data.x_coordinates.append(x_data)  
        airfoil_data.y_coordinates.append(y_data)     
        airfoil_data.x_upper_surface.append(x_up_surf_new)
        airfoil_data.x_lower_surface.append(x_lo_surf_new)
        airfoil_data.y_upper_surface.append(y_up_surf_new)
        airfoil_data.y_lower_surface.append(y_lo_surf_new)          
        airfoil_data.camber_coordinates.append(camber)

    return airfoil_data 